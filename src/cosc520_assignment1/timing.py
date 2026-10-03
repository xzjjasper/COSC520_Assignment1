"""Wall-clock timing with explicit return values and no logging inside timers."""
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from time import perf_counter_ns
from typing import Generic, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class TimedResult(Generic[T]):
    """Original result and elapsed seconds for one successful call."""
    value: T
    elapsed_seconds: float


def timed(function: Callable[P, T]) -> Callable[P, TimedResult[T]]:
    """Decorate a callable; preserve arguments/metadata and propagate exceptions."""
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TimedResult[T]:
        """Measure one call using a monotonic nanosecond clock."""
        start = perf_counter_ns()
        value = function(*args, **kwargs)
        elapsed = perf_counter_ns() - start
        return TimedResult(value, elapsed / 1_000_000_000)
    return wrapper
