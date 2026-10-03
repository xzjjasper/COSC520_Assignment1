"""Measured searches, retained structure size and explicit failed-run records.

Lookup timings include the Python loop/result list but exclude data loading,
warmup, validation and memory measurement. Build is end-to-end construction,
including streaming/decoding the input when using a disk-backed dataset.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import random
import sys
from typing import Literal

import psutil

from . import BloomFilter, CuckooFilter, HashTable, LinearSearch, SortedArray
from .bloom_filter import bloom_parameters
from .datasets import LoginDataset, generate_dataset, load_dataset, subset_dataset
from .interfaces import LoginChecker
from .timing import timed

Algorithm = Literal["linear", "binary", "hash", "bloom", "cuckoo"]
Operation = Literal["build", "lookup_hit", "lookup_miss"]
ALGORITHMS: tuple[Algorithm, ...] = ("linear", "binary", "hash", "bloom", "cuckoo")
DEFAULT_SIZES = (1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Experiment defaults through 100 million keys; reuse the master file by prefix."""
    sizes: tuple[int, ...] = DEFAULT_SIZES
    algorithms: tuple[Algorithm, ...] = ALGORITHMS
    repeats: int = 3
    workers: int = 2
    queries_per_kind: int = 1000
    warmup_queries: int = 100
    query_batch_size: int = 64
    seed: int = 42
    login_length: int = 7
    bloom_false_positive_rate: float = 0.01
    cuckoo_bucket_size: int = 4
    cuckoo_fingerprint_bits: int = 8
    cuckoo_max_kicks: int = 500
    cuckoo_target_load: float = 0.90
    memory_budget_bytes: int = 12 * 1024**3

    def validate(self) -> None:
        """Reject unsupported experiment settings before constructing anything."""
        positive = (*self.sizes, self.repeats, self.queries_per_kind,
                    self.query_batch_size, self.workers, self.cuckoo_bucket_size,
                    self.cuckoo_max_kicks, self.memory_budget_bytes)
        if not self.sizes or any(type(x) is not int or x <= 0 for x in positive):
            raise ValueError("sizes, counts and memory budget must be positive integers")
        if len(set(self.sizes)) != len(self.sizes):
            raise ValueError("sizes must not repeat")
        if not self.algorithms or len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("choose at least one algorithm, without duplicates")
        if any(a not in ALGORITHMS for a in self.algorithms):
            raise ValueError("unknown algorithm")
        if type(self.warmup_queries) is not int or self.warmup_queries < 0:
            raise ValueError("warmup_queries must be a nonnegative integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0,2**32) for PYTHONHASHSEED compatibility")
        if self.login_length not in (7, 8):
            raise ValueError("login_length must be 7 or 8")
        if not 0 < self.bloom_false_positive_rate < 1:
            raise ValueError("Bloom target probability must be in (0,1)")
        if not 0 < self.cuckoo_target_load < 1:
            raise ValueError("Cuckoo target load must be in (0,1)")
        if type(self.cuckoo_fingerprint_bits) is not int or not 1 <= self.cuckoo_fingerprint_bits <= 8:
            raise ValueError("the current bytearray Cuckoo implementation supports 1..8 bits")


@dataclass(frozen=True)
class BenchmarkRecord:
    """One repetition/category. None denotes unavailable, never a zero timing.

    Categories are ground truth: a filter false positive is still a lookup_miss.
    structure_bytes is retained Python storage after successful construction;
    failed/skipped builds have no size. Invalid searches keep diagnostics but
    are excluded from all successful-performance plots.
    """
    algorithm: Algorithm
    n: int
    repeat: int
    operation: Operation
    operations: int
    elapsed_seconds: float | None
    false_positives: int | None
    false_negatives: int | None
    structure_bytes: int | None = None
    status: Literal["ok", "error", "skipped"] = "ok"
    error: str = ""
    worker_pid: int | None = None


def _bucket_count(n: int, config: BenchmarkConfig) -> int:
    """Choose power-of-two capacity at or below the requested Cuckoo load."""
    minimum = max(2, math.ceil(n / config.cuckoo_bucket_size / config.cuckoo_target_load))
    return 1 << (minimum - 1).bit_length()


def build_checker(algorithm: Algorithm, logins: Sequence[str], config: BenchmarkConfig) -> LoginChecker:
    """Construct the selected class; insertion errors propagate."""
    if algorithm == "linear":
        return LinearSearch(logins)
    if algorithm == "binary":
        return SortedArray(logins)
    if algorithm == "hash":
        return HashTable(logins)
    if algorithm == "bloom":
        return BloomFilter(len(logins), logins=logins, seed=config.seed,
                           false_positive_rate=config.bloom_false_positive_rate)
    if algorithm == "cuckoo":
        return CuckooFilter(bucket_count=_bucket_count(len(logins), config),
                            bucket_size=config.cuckoo_bucket_size,
                            fingerprint_bits=config.cuckoo_fingerprint_bits,
                            max_kicks=config.cuckoo_max_kicks, seed=config.seed, logins=logins)
    raise ValueError(f"unknown algorithm: {algorithm}")


def query_batch(checker: LoginChecker, queries: Sequence[str]) -> list[bool]:
    """Preserve query order; leave validation outside this measured function."""
    return [checker.contains(login) for login in queries]


def measure_structure_bytes(checker: LoginChecker) -> int:
    """Count retained instance/containers/keys via sys.getsizeof, not process RSS.

    For the five known classes only, with this benchmark's unique insertion keys.
    Unique key strings and independently allocated bytearray buffers are counted
    without an O(n) set of their identities. Other objects are identity-deduplicated. References to
    shared input strings are included; external datasets, classes and methods
    are excluded. This is Python storage, not theoretical fingerprint/bit size
    or peak construction memory. Do not use on arbitrary object graphs.
    """
    if not isinstance(checker, (LinearSearch, SortedArray, HashTable, BloomFilter, CuckooFilter)):
        raise TypeError("unsupported checker type")
    seen: set[int] = set()

    def retained(value: object) -> int:
        if isinstance(value, (str, bytearray)):
            return sys.getsizeof(value)
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        total = sys.getsizeof(value)
        if isinstance(value, dict):
            total += sum(retained(k) + retained(v) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            total += sum(retained(item) for item in value)
        elif hasattr(value, "__dict__"):
            total += retained(vars(value))
        return total

    return retained(checker)


def estimate_build_bytes(algorithm: Algorithm, n: int, config: BenchmarkConfig) -> int:
    """Conservative planning estimate, not a measured size or hard memory limit."""
    string_bytes = sys.getsizeof("x" * config.login_length)
    overhead = 16 * 1024**2
    if algorithm in ("linear", "binary", "hash"):
        # Include transient dict.fromkeys storage used by the current array classes.
        extra = 64 if algorithm in ("linear", "binary") else 32
        return overhead + n * (string_bytes + extra)
    if algorithm == "bloom":
        bits = -n * math.log(config.bloom_false_positive_rate) / math.log(2)**2
        return overhead + math.ceil(bits / 8) * 2
    buckets = _bucket_count(n, config)
    # Include list pointers, allocator rounding and allocation slack per bucket.
    return overhead + buckets * (sys.getsizeof(bytearray(config.cuckoo_bucket_size)) + 32)


def _query_schedule(dataset: LoginDataset, config: BenchmarkConfig, repeat: int) -> list[tuple[str, tuple[str, ...]]]:
    """Shuffle both categories and their batches identically for every algorithm."""
    rng = random.Random(config.seed + repeat)
    batches = []
    for category, source in (("lookup_hit", dataset.hits), ("lookup_miss", dataset.misses)):
        queries = list(source)
        rng.shuffle(queries)
        batches.extend((category, tuple(queries[i:i + config.query_batch_size]))
                       for i in range(0, len(queries), config.query_batch_size))
    rng.shuffle(batches)
    return batches


def _measure_one(dataset: LoginDataset, config: BenchmarkConfig, algorithm: Algorithm,
                 repeat: int, memory_budget: int) -> list[BenchmarkRecord]:
    """Build/query/release one structure entirely inside its assigned process."""
    n = len(dataset.logins)
    for queries in (dataset.hits, dataset.misses):
        for login in queries:
            hash(login)
    schedule = _query_schedule(dataset, config, repeat)
    records: list[BenchmarkRecord] = []
    timed_build, timed_queries = timed(build_checker), timed(query_batch)
    estimate = estimate_build_bytes(algorithm, n, config)
    # The parent has reserved this job against the shared batch memory budget.
    budget = memory_budget
    if estimate > budget:
        records.append(BenchmarkRecord(algorithm, n, repeat, "build", n, None, None, None,
                                       status="skipped", error=f"estimated build {estimate:,} bytes exceeds budget {budget:,}"))
        return records
    checker = None
    stage: Operation = "build"
    build_index = len(records)
    structure_bytes = None
    try:
        built = timed_build(algorithm, dataset.logins, config)
        checker, build_seconds = built.value, built.elapsed_seconds
        del built
        # Detect known invalid states without modifying the student class.
        # In particular, a full linear-probing table can hang on a miss.
        if isinstance(checker, HashTable) and (not checker._table or None not in checker._table):
            raise ValueError("HashTable has no empty slot; absent-key probing may never terminate")
        if isinstance(checker, BloomFilter) and (checker._bit_count <= 0 or checker._hash_count <= 0):
            raise ValueError("BloomFilter constructed with nonpositive bit/hash counts")
        structure_bytes = measure_structure_bytes(checker)
        records.append(BenchmarkRecord(algorithm, n, repeat, "build", n, build_seconds,
                                       None, None, structure_bytes))
        stage = "lookup_hit"
        query_batch(checker, dataset.hits[:config.warmup_queries])
        stage = "lookup_miss"
        query_batch(checker, dataset.misses[:config.warmup_queries])
        totals = {"lookup_hit": [0.0, 0, 0, 0], "lookup_miss": [0.0, 0, 0, 0]}
        for category, queries in schedule:
            stage = category
            measured = timed_queries(checker, queries)
            answers = measured.value
            if any(type(answer) is not bool for answer in answers):
                raise TypeError("contains must return bool")
            counts = totals[category]
            counts[0] += measured.elapsed_seconds
            counts[1] += len(queries)
            counts[2] += sum(answers) if category == "lookup_miss" else 0
            counts[3] += len(answers) - sum(answers) if category == "lookup_hit" else 0
        invalid = any(v[3] for v in totals.values()) or (
            algorithm in ("linear", "binary", "hash") and totals["lookup_miss"][2] > 0)
        reason = "membership validation failed: false negatives or exact-structure false positives" if invalid else ""
        if invalid:
            records[build_index] = BenchmarkRecord(algorithm, n, repeat, "build", n, build_seconds,
                                                   None, None, structure_bytes, "error", reason)
        for category, (seconds, count, fp, fn) in totals.items():
            records.append(BenchmarkRecord(algorithm, n, repeat, category, int(count), float(seconds),
                                           int(fp), int(fn), structure_bytes, "error" if invalid else "ok", reason))
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        # Exclude the whole repetition if search failed after construction.
        if len(records) > build_index:
            previous = records[build_index]
            records[build_index] = BenchmarkRecord(**{**asdict(previous), "status": "error", "error": reason})
        records.append(BenchmarkRecord(algorithm, n, repeat, stage,
                                       n if stage == "build" else len(dataset.hits if stage == "lookup_hit" else dataset.misses),
                                       None, None, None, structure_bytes, "error", reason))
    finally:
        checker = None
        gc.collect()
    return records


def _benchmark_one(dataset: LoginDataset, config: BenchmarkConfig, algorithm: Algorithm,
                   repeat: int, memory_budget: int) -> list[BenchmarkRecord]:
    """Picklable process entry point; tag measurements with the executing PID."""
    return [replace(record, worker_pid=os.getpid()) for record in
            _measure_one(dataset, config, algorithm, repeat, memory_budget)]


def benchmark_dataset(dataset: LoginDataset, config: BenchmarkConfig,
                      *, progress: Callable[[str], None] | None = None,
                      executor: Executor | None = None,
                      checkpoint: Callable[[Sequence[BenchmarkRecord]], None] | None = None,
                      ) -> list[BenchmarkRecord]:
    """Run up to workers independent jobs, reserving their combined estimated RAM.

    A large job can use the shared budget alone rather than being rejected merely
    because two workers are configured. Small jobs may run alongside it when their
    summed estimates fit. Estimates are planning bounds, not OS-enforced limits.
    """
    config.validate()
    n = len(dataset.logins)
    if n <= 0 or not dataset.hits or not dataset.misses:
        raise ValueError("nonempty insertions and both query categories are required")
    if config.workers > 1 and executor is None:
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=get_context("spawn")) as pool:
            return benchmark_dataset(dataset, config, progress=progress, executor=pool, checkpoint=checkpoint)
    shared_budget = int(psutil.virtual_memory().available * 0.7)
    job_budget = min(config.memory_budget_bytes, shared_budget)
    jobs = []
    for repeat in range(config.repeats):
        algorithms = list(config.algorithms)
        random.Random(config.seed + repeat + n).shuffle(algorithms)
        jobs.extend((algorithm, repeat) for algorithm in algorithms)
    records = []

    def completed(result: list[BenchmarkRecord], algorithm: Algorithm, repeat: int) -> None:
        records.extend(result)
        if checkpoint is not None:
            checkpoint(records)
        if progress:
            progress(f"Finished {algorithm} n={n:,} repetition={repeat + 1}: {result[0].status} (pid={result[0].worker_pid})")

    if progress:
        progress(f"n={n:,}: {len(jobs)} jobs, at most {config.workers} workers; "
                 f"shared RAM budget={shared_budget:,}, per-job cap={job_budget:,} bytes")
    if executor is None:
        for algorithm, repeat in jobs:
            completed(_benchmark_one(dataset, config, algorithm, repeat, job_budget), algorithm, repeat)
    else:
        waiting = []
        for algorithm, repeat in jobs:
            estimate = estimate_build_bytes(algorithm, n, config)
            if estimate > job_budget:
                result = [BenchmarkRecord(algorithm, n, repeat, "build", n, None, None, None,
                          status="skipped", error=f"estimated build {estimate:,} bytes exceeds budget {job_budget:,}")]
                completed(result, algorithm, repeat)
            else:
                waiting.append((algorithm, repeat, estimate))
        pending = {}
        reserved = 0
        while waiting or pending:
            while len(pending) < config.workers:
                candidate = next((i for i, (_, _, estimate) in enumerate(waiting)
                                  if reserved + estimate <= shared_budget), None)
                if candidate is None:
                    break
                algorithm, repeat, estimate = waiting.pop(candidate)
                future = executor.submit(_benchmark_one, dataset, config, algorithm, repeat, job_budget)
                pending[future] = (algorithm, repeat, estimate)
                reserved += estimate
                if progress:
                    progress(f"Started {algorithm} n={n:,} repetition={repeat + 1}; "
                             f"active jobs={len(pending)}, reserved={reserved:,} bytes")
            if not pending:
                raise RuntimeError("No job can be scheduled within the shared budget")
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                algorithm, repeat, estimate = pending.pop(future)
                reserved -= estimate
                try:
                    result = future.result()
                except Exception as exc:
                    result = [BenchmarkRecord(algorithm, n, repeat, "build", n, None, None, None,
                              status="error", error=f"worker failure: {type(exc).__name__}: {exc}")]
                completed(result, algorithm, repeat)
    order = {"build": 0, "lookup_hit": 1, "lookup_miss": 2}
    return sorted(records, key=lambda r: (r.repeat, ALGORITHMS.index(r.algorithm), order[r.operation]))


def environment_metadata(config: BenchmarkConfig, *, data_dir: Path | None = None) -> dict:
    """Capture parameters and provenance needed to interpret measured results."""
    package_dir = Path(__file__).parent
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package_dir.glob("*.py"))}
    data = None
    if data_dir is not None:
        manifest = data_dir / "manifest.json"
        data = {"directory": str(data_dir.resolve()), "manifest": json.loads(manifest.read_text())}
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "physical_ram_bytes": psutil.virtual_memory().total,
        "available_ram_bytes_at_metadata": psutil.virtual_memory().available,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unfixed (process randomization)"),
        "source_sha256": sources, "dataset": data,
        "parallelism": {"workers": config.workers, "mode": "spawn processes" if config.workers > 1 else "serial",
                        "memory_budget": "per-job cap plus summed reservations limited to 70% of available RAM at each size; large jobs may run alone",
                        "timing_note": "parallel timings include competition for CPU/cache/memory bandwidth; use workers=1 for isolated comparisons"},
        "parameters_by_size": {str(n): {"cuckoo_bucket_count": _bucket_count(n, config),
            "cuckoo_actual_load": n / (_bucket_count(n, config) * config.cuckoo_bucket_size),
            "bloom_bits": bloom_parameters(n, config.bloom_false_positive_rate)[0],
            "bloom_hashes": bloom_parameters(n, config.bloom_false_positive_rate)[1]} for n in config.sizes},
        "lookup_timing": "perf_counter_ns around contains loop and result-list allocation; shuffled same-category batches; excluded setup/warmup/validation/I/O/size measurement",
        "build_timing": "end-to-end constructor including input iteration and disk decoding/generation, allocation and sorting/insertion",
        "size_definition": "retained Python object size including containers and unique stored strings; excludes external input/query data, classes, peak memory and process RSS",
        "hash_cache": "All query strings are prehashed outside timing, standardizing Python's cached string-hash state; timings describe steady-state lookup, not first hashing of freshly decoded strings",
        "correctness_scope": "all timed queries; not exhaustive validation of every inserted key",
        "dataset_checksum_policy": "insertion file size and deterministic samples checked at load; full SHA256 requires explicit verify_checksum=True",
    }


def run_benchmarks(config: BenchmarkConfig, *, data_dir: Path | None = None,
                   output_dir: Path | None = None,
                   progress: Callable[[str], None] | None = None) -> list[BenchmarkRecord]:
    """Reuse a saved master input by prefix; export CSV/metadata if requested.

    Without data_dir, generate a lazy master in memory. Output directories must
    be new; partial runs retain an .incomplete marker, never overwrite old work.
    """
    config.validate()
    master = (load_dataset(data_dir) if data_dir is not None else
              generate_dataset(max(config.sizes), config.queries_per_kind,
                               seed=config.seed, login_length=config.login_length))
    if max(config.sizes) > len(master.logins):
        raise ValueError("requested size exceeds the master insertion dataset")
    if master.login_length != config.login_length:
        raise ValueError("config login_length differs from the master dataset")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / ".incomplete").touch()
        (output_dir / "environment.json").write_text(json.dumps(environment_metadata(config, data_dir=data_dir), indent=2) + "\n")
    records = []

    def checkpoint(current: Sequence[BenchmarkRecord]) -> None:
        """Atomically retain completed jobs during a long run; marker stays incomplete."""
        if output_dir is not None:
            temporary = output_dir / "partial.next.csv"
            save_results([*records, *current], temporary)
            temporary.replace(output_dir / "partial.csv")

    pool = (ProcessPoolExecutor(max_workers=config.workers, mp_context=get_context("spawn"))
            if config.workers > 1 else nullcontext(None))
    with pool as executor:
        for n in config.sizes:
            dataset = subset_dataset(master, n, queries_per_kind=config.queries_per_kind, seed=config.seed)
            if output_dir is not None:
                query_dir = output_dir / f"queries-{n}"
                query_dir.mkdir()
                (query_dir / "hits.txt").write_text("\n".join(dataset.hits) + "\n", encoding="ascii")
                (query_dir / "misses.txt").write_text("\n".join(dataset.misses) + "\n", encoding="ascii")
            records.extend(benchmark_dataset(dataset, config, progress=progress, executor=executor, checkpoint=checkpoint))
    if output_dir is not None:
        save_results(records, output_dir / "raw.csv")
        metadata_path = output_dir / "environment.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["worker_pids"] = sorted({r.worker_pid for r in records if r.worker_pid is not None})
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        (output_dir / "partial.csv").unlink(missing_ok=True)
        (output_dir / ".incomplete").unlink()
    return records


def _validate_record(record: BenchmarkRecord) -> None:
    """Validate CSV semantics before saving, plotting or accepting external rows."""
    if record.algorithm not in ALGORITHMS or record.operation not in ("build", "lookup_hit", "lookup_miss"):
        raise ValueError("unknown algorithm/operation")
    if record.status not in ("ok", "error", "skipped"):
        raise ValueError("unknown status")
    if record.n <= 0 or record.repeat < 0 or record.operations <= 0:
        raise ValueError("invalid sizes/counts")
    if record.elapsed_seconds is not None and (not math.isfinite(record.elapsed_seconds) or record.elapsed_seconds < 0):
        raise ValueError("invalid elapsed time")
    if record.status == "ok" and (record.elapsed_seconds is None or record.error):
        raise ValueError("successful records need a time and no error")
    if record.status != "ok" and not record.error:
        raise ValueError("unsuccessful records need an explanation")
    if record.worker_pid is not None and record.worker_pid <= 0:
        raise ValueError("worker_pid must be positive")
    if record.structure_bytes is not None and record.structure_bytes <= 0:
        raise ValueError("structure_bytes must be positive when available")
    for value in (record.false_positives, record.false_negatives):
        if value is not None and not 0 <= value <= record.operations:
            raise ValueError("error count out of range")
    if record.operation == "build":
        if record.operations != record.n or record.false_positives is not None or record.false_negatives is not None:
            raise ValueError("invalid build record")
    elif record.status == "ok":
        if record.false_positives is None or record.false_negatives is None:
            raise ValueError("successful queries require correctness counts")
        if record.false_negatives or (record.operation == "lookup_hit" and record.false_positives):
            raise ValueError("successful queries cannot contain false negatives/hit false positives")
        if record.algorithm in ("linear", "binary", "hash") and record.false_positives:
            raise ValueError("exact structures cannot have successful false positives")


def save_results(records: Sequence[BenchmarkRecord], path: Path) -> None:
    """Write every raw repetition; fail if the CSV exists, preserve None as blank."""
    for record in records:
        _validate_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(BenchmarkRecord)])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def load_results(path: Path) -> list[BenchmarkRecord]:
    """Load the current schema; reject malformed/incomplete rows and invalid data."""
    records = []
    names = [field.name for field in fields(BenchmarkRecord)]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (names, names[:-1]):
            raise ValueError("unexpected CSV schema")
        for number, row in enumerate(reader, start=2):
            try:
                row.setdefault("worker_pid", "")
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("missing or extra CSV cells")
                for key in ("n", "repeat", "operations"):
                    row[key] = int(row[key])
                for key in ("false_positives", "false_negatives", "structure_bytes", "worker_pid"):
                    row[key] = int(row[key]) if row[key] else None
                row["elapsed_seconds"] = float(row["elapsed_seconds"]) if row["elapsed_seconds"] else None
                record = BenchmarkRecord(**row)
                _validate_record(record)
                records.append(record)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid CSV row {number}: {exc}") from exc
    return records
