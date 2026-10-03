from collections.abc import Iterable, Sized
import math

class HashTable:
    """Exact membership using manually managed table and collision handling."""
    _table: list[str|None]
    _size: int
    _max_load_factor: float

    def __init__(
        self,
        logins: Iterable[str] = (),
        *,
        max_load_factor: float = 0.75,
    ) -> None:
        """Input: initial strings and load threshold. Output: None; initialize the table.
        Reuse sized iterables directly. Materialize a one-shot iterator once, so
        counting does not consume the keys before insertion. Inputs/load factor
        are assumed valid; this fixed-capacity table does not implement resizing.
        """
        if not isinstance(logins, Sized):
            logins = list(logins)
        self._size = 0
        self._max_load_factor = max_load_factor
        self._table = [None] * int(len(logins) // max_load_factor + 1)
        for login in logins:
            self.add(login)

    def add(self, login: str) -> None:
        """Input: login string. Output: None on successful or duplicate insertion.
        Linear probing ignores an existing key. The current capacity policy raises
        after a new insertion reaches the threshold; it does not undo that insertion.
        """
        # Calculate the index for the given login
        index = hash(login) % len(self._table)

        # Handle collisions by linear probing
        while self._table[index] is not None:
            if self._table[index] == login:
                return  # Login already exists
            index = (index + 1) % len(self._table)

        # Insert the login at the found index
        self._table[index] = login
        self._size += 1

        if self._size >= len(self._table) * self._max_load_factor:
            raise AssertionError("Reached max load factor; resizing not implemented.")

    def contains(self, login: str) -> bool:
        """Input: query string. Output: exact membership boolean.
        Probe until a matching key or empty slot. A full traversal raises under
        the fixed-capacity policy. Normal initial loading retains an empty slot.
        """
        index = hash(login) % len(self._table)
        start_index = index
        # Search for the login using linear probing
        while self._table[index] is not None:
            if self._table[index] == login:
                return True  # Login found
            index = (index + 1) % len(self._table)
            if index == start_index:
                raise AssertionError("No spare slot found; table is full. Resizing not implemented.")

        return False  # Login not found
