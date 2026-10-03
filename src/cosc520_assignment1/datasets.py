"""Unique synthetic usernames, bounded-memory export and disk-backed sequences.

Generator v2 permutes the finite base-62 namespace with four alternating Feistel
updates, then encodes the result as exactly 7 or 8 ASCII characters. Every update
is invertible, so distinct IDs cannot collide. Miss IDs lie beyond the master set.
This is synthetic benchmark data, not a cryptographic username generator.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import json
import operator
from pathlib import Path
import random
import shutil
import time

import numpy as np

MASK = (1 << 64) - 1
GENERATOR = "base62-feistel-v2"
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
ROUND_CONSTANT = 0x9E3779B97F4A7C15


def _mix64(value: int) -> int:
    """Map a uint64 to a uint64 bijectively; used only to generate input data."""
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)


def login_for_index(index: int, seed: int = 42, login_length: int = 7) -> str:
    """Encode a unique ID via a bijection of the 62**width namespace.

    Four alternating modular Feistel updates scramble the insertion order without
    collision rejection or a giant uniqueness set. This is synthetic data, not a
    cryptographic cipher. Each update is inverted by subtracting the same function.
    """
    if login_length not in (7, 8):
        raise ValueError("login_length must be 7 or 8")
    if not 0 <= index < 62**login_length or not 0 <= seed <= MASK:
        raise ValueError("index must fit the base-62 namespace; seed must be uint64")
    left_mod = 62 ** (login_length // 2)
    right_mod = 62 ** (login_length - login_length // 2)
    left, right = divmod(index, right_mod)
    for round_number in range(4):
        key = (seed + (round_number + 1) * ROUND_CONSTANT) & MASK
        if round_number % 2 == 0:
            left = (left + _mix64(right ^ key) % left_mod) % left_mod
        else:
            right = (right + _mix64(left ^ key) % right_mod) % right_mod
    value = left * right_mod + right
    characters = ["0"] * login_length
    for position in range(login_length - 1, -1, -1):
        value, digit = divmod(value, 62)
        characters[position] = ALPHABET[digit]
    return "".join(characters)


@dataclass(frozen=True)
class GeneratedLogins(Sequence[str]):
    """Lazy insertion sequence; len/index/iteration do not materialize a list."""
    size: int
    seed: int
    login_length: int

    def __len__(self) -> int:
        """Return the number of insertion records."""
        return self.size

    def __getitem__(self, index: int | slice) -> str | list[str]:
        """Return one username or an explicitly requested materialized slice."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.size))]
        index = operator.index(index)
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        return login_for_index(index, self.seed, self.login_length)

    def __iter__(self) -> Iterator[str]:
        """Yield input keys in reproducible, non-lexicographic order."""
        for i in range(self.size):
            yield login_for_index(i, self.seed, self.login_length)


@dataclass(frozen=True)
class FileLogins(Sequence[str]):
    """Fixed-width ASCII records with one LF each; no billion-object Python list."""
    path: Path
    size: int
    login_length: int

    def __len__(self) -> int:
        """Return the visible prefix length."""
        return self.size

    def __getitem__(self, index: int | slice) -> str | list[str]:
        """Read one record by seek; slices intentionally materialize only that slice."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.size))]
        index = operator.index(index)
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        with self.path.open("rb") as stream:
            stream.seek(index * (self.login_length + 1))
            raw = stream.read(self.login_length + 1)
        if len(raw) != self.login_length + 1 or raw[-1:] != b"\n":
            raise ValueError("Truncated or malformed insertion file")
        return raw[:-1].decode("ascii")

    def __iter__(self) -> Iterator[str]:
        """Read sequentially with bounded buffering; stop at the visible prefix."""
        with self.path.open("rb", buffering=4 * 1024 * 1024) as stream:
            for _ in range(self.size):
                raw = stream.readline(self.login_length + 2)
                if len(raw) != self.login_length + 1 or raw[-1:] != b"\n":
                    raise ValueError("Truncated or malformed insertion file")
                yield raw[:-1].decode("ascii")


@dataclass(frozen=True)
class LoginDataset:
    """Insertion sequence and queries labeled by true membership, not filter output."""
    logins: Sequence[str]
    hits: tuple[str, ...]
    misses: tuple[str, ...]
    seed: int
    login_length: int
    universe_size: int


def generate_dataset(n: int, queries_per_kind: int = 10_000, *, seed: int = 42,
                     login_length: int = 7) -> LoginDataset:
    """Create lazy unique insertions and equal hit/miss query lists; no disk writes."""
    if not isinstance(n, int) or isinstance(n, bool) or not 0 < n < 62**login_length:
        raise ValueError("n must be positive and leave unused base-62 IDs for misses")
    if not isinstance(queries_per_kind, int) or queries_per_kind <= 0:
        raise ValueError("queries_per_kind must be positive")
    login_for_index(0, seed, login_length)
    return subset_dataset(
        LoginDataset(GeneratedLogins(n, seed, login_length), (), (), seed, login_length, n),
        n, queries_per_kind=queries_per_kind, seed=seed,
    )


def subset_dataset(dataset: LoginDataset, n: int, *, queries_per_kind: int = 1000,
                   seed: int = 42) -> LoginDataset:
    """View the first n insertions; misses remain absent from the entire master set."""
    if not 0 < n <= len(dataset.logins) or queries_per_kind <= 0:
        raise ValueError("Invalid prefix size or query count")
    rng = random.Random(seed)
    hit_ids = ([rng.randrange(n) for _ in range(queries_per_kind)] if n < queries_per_kind
               else rng.sample(range(n), queries_per_kind))
    namespace = 62**dataset.login_length
    if namespace - dataset.universe_size < queries_per_kind:
        raise ValueError("Not enough unused IDs for unique miss queries")
    ordered_misses = rng.sample(range(dataset.universe_size, namespace), queries_per_kind)
    if isinstance(dataset.logins, GeneratedLogins):
        logins = GeneratedLogins(n, dataset.seed, dataset.login_length)
    elif isinstance(dataset.logins, FileLogins):
        logins = FileLogins(dataset.logins.path, n, dataset.login_length)
    else:
        raise TypeError("subset_dataset requires a generated or file-backed master dataset")
    return LoginDataset(
        logins,
        tuple(login_for_index(i, dataset.seed, dataset.login_length) for i in hit_ids),
        tuple(login_for_index(i, dataset.seed, dataset.login_length) for i in ordered_misses),
        dataset.seed, dataset.login_length, dataset.universe_size,
    )


def _encoded_chunk(start: int, stop: int, seed: int, width: int) -> bytes:
    """Vectorize the same bijection as login_for_index, returning ASCII lines."""
    def mix(values: np.ndarray) -> np.ndarray:
        """Vectorized uint64 wraparound permutation for bounded export chunks."""
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> 27)) * np.uint64(0x94D049BB133111EB)
        return values ^ (values >> 31)

    ids = np.arange(start, stop, dtype=np.uint64)
    left_mod = np.uint64(62 ** (width // 2))
    right_mod = np.uint64(62 ** (width - width // 2))
    left, right = ids // right_mod, ids % right_mod
    for round_number in range(4):
        key = np.uint64((seed + (round_number + 1) * ROUND_CONSTANT) & MASK)
        if round_number % 2 == 0:
            left = (left + mix(right ^ key) % left_mod) % left_mod
        else:
            right = (right + mix(left ^ key) % right_mod) % right_mod
    values = left * right_mod + right
    alphabet = np.frombuffer(ALPHABET.encode("ascii"), dtype=np.uint8)
    output = np.empty((stop - start, width + 1), dtype=np.uint8)
    for position in range(width - 1, -1, -1):
        output[:, position] = alphabet[values % np.uint64(62)]
        values //= np.uint64(62)
    output[:, width] = 10
    return output.tobytes()


def save_dataset(dataset: LoginDataset, directory: Path, *, chunk_size: int = 1_000_000,
                 progress: Callable[[int, int], None] | None = None) -> Path:
    """Write all insertion records and labeled queries with checksums, bounded in RAM.

    Existing destinations are never overwritten. A manifest is published only
    after every file completes; interrupted exports retain an .incomplete marker.
    """
    directory = Path(directory)
    if chunk_size <= 0 or not isinstance(dataset.logins, GeneratedLogins):
        raise ValueError("Export requires GeneratedLogins and a positive chunk_size")
    directory.parent.mkdir(parents=True, exist_ok=True)
    needed = (len(dataset.logins) + len(dataset.hits) + len(dataset.misses)) * (dataset.login_length + 1)
    if shutil.disk_usage(directory.parent).free < needed + 2 * 1024**3:
        raise OSError(f"Need {needed:,} bytes plus 2 GiB free reserve")
    directory.mkdir(exist_ok=False)
    marker = directory / ".incomplete"
    marker.write_text("Export incomplete: do not benchmark this directory.\n")
    files = {}
    digest = hashlib.sha256()
    path = directory / "logins.txt"
    with path.open("wb", buffering=4 * 1024 * 1024) as stream:
        for start in range(0, len(dataset.logins), chunk_size):
            stop = min(start + chunk_size, len(dataset.logins))
            block = _encoded_chunk(start, stop, dataset.seed, dataset.login_length)
            stream.write(block)
            digest.update(block)
            if progress is not None:
                progress(stop, len(dataset.logins))
    files[path.name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    for filename, rows in (("hits.txt", dataset.hits), ("misses.txt", dataset.misses)):
        raw = ("\n".join(rows) + "\n").encode("ascii")
        (directory / filename).write_bytes(raw)
        files[filename] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    manifest = {
        "version": 2, "generator": GENERATOR, "created_unix": time.time(),
        "n": len(dataset.logins), "universe_size": dataset.universe_size,
        "queries_per_kind": len(dataset.hits), "seed": dataset.seed,
        "login_length": dataset.login_length, "encoding": "ascii", "newline": "LF",
        "alphabet": ALPHABET, "namespace_size": 62**dataset.login_length,
        "uniqueness": "four invertible alternating modular Feistel updates on base-62 IDs",
        "insertion_ids": [0, len(dataset.logins)], "miss_ids": "outside [0, universe_size)",
        "files": files,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    marker.unlink()
    return manifest_path


def load_dataset(directory: Path, *, verify_checksum: bool = False) -> LoginDataset:
    """Open disk-backed insertions; validate sizes/query hashes and optionally scan all bytes.

    Full checksum validation reads the entire file (8 GB for one billion seven-character keys).
    Default loading checks sizes, all query checksums, and deterministic record samples.
    """
    directory = Path(directory)
    if (directory / ".incomplete").exists():
        raise ValueError("Dataset export is incomplete")
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != 2 or manifest["generator"] != GENERATOR:
        raise ValueError("Unsupported dataset format")
    n, width, seed = manifest["n"], manifest["login_length"], manifest["seed"]
    if not 0 < n <= manifest["universe_size"] < 62**width:
        raise ValueError("Invalid insertion count in manifest")
    login_for_index(0, seed, width)
    for filename in ("logins.txt", "hits.txt", "misses.txt"):
        meta = manifest["files"][filename]
        path = directory / filename
        expected_count = n if filename == "logins.txt" else manifest["queries_per_kind"]
        if path.stat().st_size != meta["bytes"] or meta["bytes"] != expected_count * (width + 1):
            raise ValueError(f"Wrong file size: {filename}")
        if verify_checksum or filename != "logins.txt":
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != meta["sha256"]:
                raise ValueError(f"Checksum mismatch: {filename}")
    logins = FileLogins(directory / "logins.txt", n, width)
    for index in {0, n // 2, n - 1}:
        if logins[index] != login_for_index(index, seed, width):
            raise ValueError("Insertion sample does not match the recorded generator")
    queries = [tuple((directory / name).read_text(encoding="ascii").splitlines())
               for name in ("hits.txt", "misses.txt")]
    if any(len(q) != manifest["queries_per_kind"] or any(len(s) != width for s in q) for q in queries):
        raise ValueError("Invalid query records")
    return LoginDataset(logins, queries[0], queries[1], seed, width, manifest["universe_size"])
