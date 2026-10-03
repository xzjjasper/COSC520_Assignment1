"""Structural typing for the student's membership implementations.

The benchmark supplies case-sensitive strings and unique insertion keys.
The protocol does not implement algorithms or runtime input validation.
"""
from typing import Protocol


class LoginChecker(Protocol):
    """Exact structures prove membership; filters indicate possible membership."""

    def add(self, login: str) -> None:
        """Insert a string; duplicate adds should not change exact membership.

        Filters can consume another slot for a repeated insertion. Cuckoo failure
        raises CuckooInsertionError and must preserve all previously accepted keys.
        """
        ...

    def contains(self, login: str) -> bool:
        """Return exact membership, or possible membership for a filter.

        Accepted keys must never produce false negatives. A filter's True alone
        does not prove that an arbitrary query was previously inserted.
        """
        ...
