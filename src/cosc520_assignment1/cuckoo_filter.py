from collections.abc import Iterable
import random

class CuckooInsertionError(RuntimeError):
    """Insertion failed without invalidating any previously accepted key."""


class CuckooFilter:
    """Fingerprints, buckets, and bounded displacement; no exact key-set backing."""
    _buckets: list[bytearray]
    _bucket_count: int
    _bucket_size: int
    _fingerprint_bits: int
    _max_kicks: int
    _seed: int

    def __init__(
        self,
        *,
        bucket_count: int = 1024,
        bucket_size: int = 4,
        fingerprint_bits: int = 8,
        max_kicks: int = 500,
        seed: int = 42,
        logins: Iterable[str] = (),
    ) -> None:
        """Input: power-of-two buckets >=2, positive bucket size/kick bound, 1..8
        fingerprint bits, seed and strings. Output: None; allocate independent bytearrays.
        Configuration is assumed valid. A private seeded generator selects candidate
        buckets and eviction slots without changing Python's global random state.
        """
        self._bucket_count = bucket_count
        self._bucket_size = bucket_size
        self._fingerprint_bits = fingerprint_bits
        self._max_kicks = max_kicks
        self._seed = seed
        self._rng = random.Random(seed)
        self._buckets = [bytearray([0] * bucket_size) for _ in range(bucket_count)]
        for login in logins:
            self.add(login)

    def _fingerprint(self, login: str) -> int:
        """Input: string. Output: nonzero f-bit integer for a valid 1..8-bit config.
        Take low hash bits, mapping zero to one to reserve empty slots. This increases
        the probability of fingerprint 1 slightly. Python string hashing is stable
        within a process; control PYTHONHASHSEED before startup for repeatable runs.
        """
        hashed = hash(login)  # Use Python's built-in hash function
        fingerprint = hashed & ((1 << self._fingerprint_bits) - 1)
        return fingerprint or 1  # Ensure the fingerprint is never zero

    def _primary_index(self, login: str) -> int:
        """Input: string. Output: bucket index in [0,bucket_count) for positive count.
        Hash the seed/key pair. The class seed alone does not control Python's process
        hash randomization; record/fix PYTHONHASHSEED for reproducible experiments.
        """
        return hash((self._seed, login)) % self._bucket_count

    def _alternate_index(self, index: int, fingerprint: int) -> int:
        """Input: current index/fingerprint. Output: the other candidate bucket.
        Hash the integer fingerprint/seed pair, XOR, then reduce to the bucket range.
        For the required power-of-two counts this is reversible.
        The two candidates can coincide; this is not itself a false-negative bug.
        """
        return (index ^ hash((fingerprint, self._seed))) % self._bucket_count

    def _insert(self, fingerprint: int, index: int) -> bool:
        """Input: fingerprint and starting bucket. Output: True if inserted, False otherwise.
        """
        for i in range(self._bucket_size):
            if self._buckets[index][i] == 0:
                self._buckets[index][i] = fingerprint
                return True
        return False

    def _relocate(self, fingerprint: int, index: int) -> None:
        """Input: fingerprint and starting bucket. Output: None or CuckooInsertionError.
        Try at most max_kicks random displacements. Revisiting a bucket is allowed:
        different slots/fingerprints can lead to a different path. Log every overwrite
        and undo it in reverse order if insertion fails, including repeated slots.
        """
        changes: list[tuple[int, int, int]] = []
        for _ in range(self._max_kicks):
            if self._insert(fingerprint, index):
                return
            slot = self._rng.randrange(self._bucket_size)
            evicted_fingerprint = self._buckets[index][slot]
            changes.append((index, slot, evicted_fingerprint))
            self._buckets[index][slot] = fingerprint
            fingerprint = evicted_fingerprint
            index = self._alternate_index(index, fingerprint)
        # The last eviction may have reached an empty slot within the kick budget.
        if self._insert(fingerprint, index):
            return
        for bucket, slot, old_fingerprint in reversed(changes):
            self._buckets[bucket][slot] = old_fingerprint
        raise CuckooInsertionError("Failed to insert fingerprint after max_kicks")

    def add(self, login: str) -> None:
        """Input: string. Output: None on success or CuckooInsertionError.
        Try an empty slot in either candidate bucket, then relocate. The dispatch logic
        preserves existing membership on failure. Repeated insertions
        may consume additional slots, consistent with fingerprint-based storage.
        """
        fingerprint = self._fingerprint(login)
        primary = self._primary_index(login)
        if self._insert(fingerprint, primary):
            return
        # first bucket is full, try alternate bucket
        alternate = self._alternate_index(primary, fingerprint)
        if self._insert(fingerprint, alternate):
            return
        # If we reach here, both buckets are full
        self._relocate(fingerprint, self._rng.choice((primary, alternate)))


    def contains(self, login: str) -> bool:
        """Input: string. Output: False=absent, True=possibly present.
        Check the fingerprint in both candidate buckets. Failed insertion rolls back
        all bucket changes, so previously accepted keys remain queryable.
        """
        fingerprint = self._fingerprint(login)
        primary = self._primary_index(login)
        alternate = self._alternate_index(primary, fingerprint)
        return fingerprint in self._buckets[primary] or fingerprint in self._buckets[alternate]
