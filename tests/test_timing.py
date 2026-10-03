import pytest
import cosc520_assignment1.timing as module


def test_decorator_forwards_and_uses_seconds(monkeypatch):
    ticks = iter([1_000_000_000, 3_500_000_000])
    monkeypatch.setattr(module, 'perf_counter_ns', lambda: next(ticks))
    @module.timed
    def calculate(x, *, y):
        """Original documentation."""
        return x + y
    result = calculate(2, y=3)
    assert result.value == 5 and result.elapsed_seconds == 2.5
    assert calculate.__name__ == 'calculate' and calculate.__doc__ == 'Original documentation.'


def test_decorator_propagates_error():
    @module.timed
    def fail():
        raise ValueError('expected')
    with pytest.raises(ValueError, match='expected'):
        fail()
