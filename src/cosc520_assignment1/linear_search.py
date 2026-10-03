from collections.abc import Iterable

class LinearSearch:
    """Exact membership via manual sequential comparisons."""
    _items: list[str]

    def __init__(self, logins: Iterable[str] = ()) -> None:
        """Input: initial strings. Output: None; materialize their insertion order.
        Initial duplicates are removed; add() ignores an existing key. Inputs are
        assumed to be strings. The benchmark supplies unique insertion keys.
        """
        self._items = list(dict.fromkeys(logins))

    def add(self, login: str) -> None:
        """Input: login string. Output: None.
        Store the key unless already present.
        """
        if not self.contains(login):
            self._items.append(login)

    def contains(self, login: str) -> bool:
        """Input: query string. Output: exact membership boolean.
        Manually scan the list.
        """
        for item in self._items:
            if item == login:
                return True
        return False
