import pytest
from matplotlib import pyplot as plt
from cosc520_assignment1.benchmark import BenchmarkRecord
from cosc520_assignment1.plotting import (
    summarize_timings, plot_runtimes, plot_storage_sizes, plot_false_positive_rates,
)


def test_summary_normalizes_each_repeat():
    records = [BenchmarkRecord('linear', 100, 0, 'lookup_hit', 10, 0.1, 0, 0),
               BenchmarkRecord('linear', 100, 1, 'lookup_hit', 20, 0.4, 0, 0)]
    summary, = summarize_timings(records)
    assert summary.median == pytest.approx(0.015)
    assert summary.q1 == pytest.approx(0.0125) and summary.q3 == pytest.approx(0.0175)


def test_failure_is_not_plotted_as_zero():
    bad = BenchmarkRecord('cuckoo', 100, 0, 'build', 100, None, None, None,
                          status='error', error='cannot build')
    assert summarize_timings([bad]) == []
    figure = plot_runtimes([bad])
    assert all(not axis.lines for axis in figure.axes)
    plt.close(figure)


def test_fpr_denominator_and_storage_units(tmp_path):
    records = [BenchmarkRecord('bloom', 100, 0, 'build', 100, 0.1, None, None, 1024**2),
               BenchmarkRecord('bloom', 100, 0, 'lookup_hit', 900, 0.1, 0, 0, 1024**2),
               BenchmarkRecord('bloom', 100, 0, 'lookup_miss', 100, 0.1, 2, 0, 1024**2)]
    rates = plot_false_positive_rates(records)
    assert rates.axes[0].lines[0].get_ydata()[0] == 2
    sizes = plot_storage_sizes(records)
    assert sizes.axes[0].lines[0].get_ydata()[0] == 1
    runtimes = plot_runtimes(records)
    assert len(runtimes.axes) == 3
    for name, figure in [('fpr', rates), ('storage', sizes), ('runtimes', runtimes)]:
        figure.savefig(tmp_path / f'{name}.png')
        assert (tmp_path / f'{name}.png').stat().st_size > 1000
        plt.close(figure)
