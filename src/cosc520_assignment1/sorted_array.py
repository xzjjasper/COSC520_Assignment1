from collections.abc import Iterable

class SortedArray:
    """Exact membership over an ascending string array."""
    _items: list[str]

    def __init__(self, logins: Iterable[str] = ()) -> None:
        """Input: initial strings. Output: None; initialize ascending storage.
        Remove duplicate initial strings with dict.fromkeys, then use list.sort().
        Binary search is implemented manually. Inputs are assumed valid strings.
        """
        self._items = list(dict.fromkeys(logins))
        self._items.sort()

    def _lower_bound(self, login: str) -> int:
        """Input: string. Output: first index with value >= login, or len(_items).
        """
        left = 0
        right = len(self._items)
        while left < right:
            mid = (left + right) // 2
            if self._items[mid] < login:
                left = mid + 1
            else:
                right = mid
        return left

    def add(self, login: str) -> None:
        """Input: string. Output: None; keep storage sorted and unique.
        """
        index = self._lower_bound(login)
        if index == len(self._items) or self._items[index] != login:
            self._items.insert(index, login)

    def contains(self, login: str) -> bool:
        """Input: query string. Output: exact membership boolean.
        """
        index = self._lower_bound(login)
        return index < len(self._items) and self._items[index] == login
