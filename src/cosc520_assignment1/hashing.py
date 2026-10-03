"""Optional deterministic utility hash, unused by the five student algorithms.

Those implementations currently use Python hash(). This helper does not replace
or fix their hashing. The assignment does not explicitly require a handwritten
hash primitive; confirm course restrictions before adopting a library primitive
inside assessed algorithms.
"""
import hashlib


def hash_string(value: str, seed: int = 0) -> int:
    """Hash UTF-8 text plus a uint64 seed to uint64, stable across processes.

    Uses standard-library BLAKE2b with an eight-byte digest and explicit seed
    prefix. Empty strings are valid; collisions are possible, as for any hash.
    Time is linear in the UTF-8 input length. This is not a membership algorithm.
    """
    if not isinstance(value, str):
        raise TypeError("value must be str")
    if type(seed) is not int:
        raise TypeError("seed must be int")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be uint64")
    digest = hashlib.blake2b(seed.to_bytes(8, "little") + value.encode("utf-8"),
                             digest_size=8, person=b"COSC520-helper").digest()
    return int.from_bytes(digest, "little")
