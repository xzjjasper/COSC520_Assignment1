"""Student Bloom filter; ordinary Bloom filters do not support deletion."""
from collections.abc import Iterable, Iterator
import math


def bloom_parameters(expected_items: int, false_positive_rate: float) -> tuple[int, int]:
    """Input: n > 0 and target rate p in (0,1). Output: (m bits, k hashes).
    Preserve the chosen ceiling rounding. Callers select parameters giving positive
    m/k; the experiment records the empirical rate separately from the target.
    """
    m = - (expected_items * math.log(false_positive_rate)) / math.log(2)**2
    k = (m / expected_items) * math.log(2)
    return math.ceil(m), math.ceil(k)


class BloomFilter:
    """Approximate membership using a packed bit array."""
    _bits: bytearray
    _bit_count: int
    _hash_count: int
    _seed: int

    def __init__(
        self,
        expected_items: int,
        *,
        false_positive_rate: float = 0.01,
        logins: Iterable[str] = (),
        seed: int = 42,
    ) -> None:
        """Input: estimated count, target rate, initial strings and seed. Output: None.
        Allocate packed bytes, then insert initial keys. The supplied configuration
        is assumed to produce positive bit/hash counts; keys are strings.
        """
        self._bit_count, self._hash_count = bloom_parameters(expected_items, false_positive_rate)
        self._bits = bytearray((self._bit_count + 7) // 8)  # Allocate enough bytes to hold m bits
        self._seed = seed
        for login in logins:
            self.add(login)

    def _hash_indexes(self, login: str) -> Iterator[int]:
        """Input: string. Output: k indexes in [0,m) for valid positive parameters.
        Tuple-hash variants can be dependent; measure the empirical false-positive
        rate instead of assuming it equals the configured target.
        Python's process hash seed also affects cross-process repeatability.
        """
        for i in range(self._hash_count):
            yield hash((login, self._seed, i)) % self._bit_count

    def _get_bit(self, index: int) -> bool:
        """Input: bit index in [0, m). Output: current bit value.
        Packed-byte addressing; reject out-of-range indexes.
        """
        if index < 0 or index >= self._bit_count:
            raise IndexError("Bit index out of range")
        byte_index = index // 8
        bit_index = index % 8
        return bool(self._bits[byte_index] & (1 << bit_index))

    def _set_bit(self, index: int) -> None:
        """Input: bit index in [0, m). Output: None.
        Set the addressed bit without changing unrelated bits; validate index.
        """
        if index < 0 or index >= self._bit_count:
            raise IndexError("Bit index out of range")
        byte_index = index // 8
        bit_index = index % 8
        self._bits[byte_index] |= 1 << bit_index

    def add(self, login: str) -> None:
        """Input: string. Output: None; update filter state.
        """
        for index in self._hash_indexes(login):
            self._set_bit(index)

    def contains(self, login: str) -> bool:
        """Input: string. Output: False=absent, True=possibly present.
        """
        for index in self._hash_indexes(login):
            if not self._get_bit(index):
                return False
        return True
