"""Plots derived only from successful measured runs; no imputed missing values."""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from .benchmark import ALGORITHMS, Algorithm, BenchmarkRecord, Operation, _validate_record

LABELS = {"linear": "Linear search", "binary": "Binary search", "hash": "Hash table",
          "bloom": "Bloom filter", "cuckoo": "Cuckoo filter"}
COLORS = {"linear": "#2463A0", "binary": "#A66D00", "hash": "#BD4E24",
          "bloom": "#617A30", "cuckoo": "#AE4D80"}
MARKERS = {"linear": "o", "binary": "s", "hash": "^", "bloom": "D", "cuckoo": "v"}


@dataclass(frozen=True)
class TimingSummary:
    """Build seconds or lookup seconds/query; IQR is repeat spread, not a CI."""
    algorithm: Algorithm
    n: int
    operation: Operation
    median: float
    q1: float
    q3: float
    samples: int


def summarize_timings(records: Sequence[BenchmarkRecord]) -> list[TimingSummary]:
    """Normalize each repetition's lookups before computing median/quartiles."""
    grouped = defaultdict(list)
    for record in records:
        _validate_record(record)
        if record.status == "ok":
            value = record.elapsed_seconds
            if record.operation != "build":
                value /= record.operations
            grouped[record.algorithm, record.n, record.operation].append(value)
    result = []
    for (algorithm, n, operation), values in sorted(grouped.items()):
        q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
        result.append(TimingSummary(algorithm, n, operation, float(median), float(q1), float(q3), len(values)))
    return result


def _finish(fig: Figure, axes, records: Sequence[BenchmarkRecord], note: str) -> None:
    """Apply shared readable styling and disclose omitted failed/skipped runs."""
    failures = sorted({(r.algorithm, r.n, r.repeat) for r in records if r.status != "ok"})
    excluded = "; ".join(f"{LABELS[a]}: {sum(item[0] == a for item in failures)}"
                         for a in ALGORITHMS if any(item[0] == a for item in failures))
    for ax in axes:
        ax.set_xlabel("Inserted usernames, n (log scale)")
        ax.set_xscale("log")
        ax.grid(True, which="major", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.text(0.5, 0.5, "No successful measurements", ha="center", va="center", transform=ax.transAxes)
    footer = note + (f"\nExcluded failed/skipped repetitions — {excluded}. See raw.csv for reasons." if excluded else "")
    fig.text(0.015, 0.018, footer, fontsize=8, color="#444444", va="bottom")
    fig.tight_layout(rect=(0, 0.15, 1, 0.96))


def _line(ax, algorithm: str, xs, median, q1, q3) -> None:
    """Draw a stable algorithm identity and a repeat-IQR band."""
    ax.plot(xs, median, label=LABELS[algorithm], color=COLORS[algorithm],
            marker=MARKERS[algorithm], linewidth=1.6,
            linestyle="--" if algorithm == "binary" else "-",
            markersize=8 if algorithm == "linear" else 5,
            markerfacecolor="none" if algorithm in ("linear", "binary") else COLORS[algorithm])
    ax.fill_between(xs, q1, q3, color=COLORS[algorithm], alpha=0.12)


def plot_runtimes(records: Sequence[BenchmarkRecord], *, log_scale: bool = True) -> Figure:
    """Three panels: end-to-end build, true-present search and true-absent search."""
    summaries = summarize_timings(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    definitions = (("build", "Load + build", "Seconds (total)", 1),
                   ("lookup_hit", "Known-present queries", "Microseconds / query", 1e6),
                   ("lookup_miss", "Known-absent queries", "Microseconds / query", 1e6))
    for ax, (operation, title, ylabel, scale) in zip(axes, definitions):
        for algorithm in ALGORITHMS:
            series = sorted((s for s in summaries if s.algorithm == algorithm and s.operation == operation), key=lambda s: s.n)
            if not series:
                continue
            if log_scale and any(s.q1 <= 0 for s in series):
                raise ValueError("logarithmic time plots require strictly positive measurements")
            _line(ax, algorithm, [s.n for s in series], [s.median * scale for s in series],
                  [s.q1 * scale for s in series], [s.q3 * scale for s in series])
        ax.set_title(title)
        ax.set_ylabel(ylabel + (" (log scale)" if log_scale else ""))
        if log_scale:
            ax.set_yscale("log")
    fig.suptitle("Measured runtime by input size", fontsize=14)
    _finish(fig, axes, records, "Median with interquartile range across successful repetitions. Search categories use ground truth, including filter false positives.\nBuild includes input iteration; search excludes setup, I/O, warmup, correctness checks and memory measurement.")
    return fig


def plot_storage_sizes(records: Sequence[BenchmarkRecord]) -> Figure:
    """Compare retained Python storage after inserting n unique keys, in MiB."""
    grouped = defaultdict(list)
    for record in records:
        _validate_record(record)
        if record.status == "ok" and record.operation == "build" and record.structure_bytes is not None:
            grouped[record.algorithm, record.n].append(record.structure_bytes / 1024**2)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    for algorithm in ALGORITHMS:
        xs = sorted(n for a, n in grouped if a == algorithm)
        if xs:
            quantiles = np.array([np.quantile(grouped[algorithm, n], (0.25, 0.5, 0.75)) for n in xs])
            _line(ax, algorithm, xs, quantiles[:, 1], quantiles[:, 0], quantiles[:, 2])
    ax.set_yscale("log")
    ax.set_ylabel("Retained Python storage, MiB (log scale)")
    ax.set_title("Structure size after loading n usernames")
    _finish(fig, [ax], records, "Median and IQR. Includes instance, containers and stored strings; excludes input/query datasets and process overhead.\nMeasured with sys.getsizeof traversal; not peak RAM or theoretical bit/fingerprint storage.")
    return fig


def plot_false_positive_rates(records: Sequence[BenchmarkRecord]) -> Figure:
    """Filter false positives divided by known-absent queries, including zeros."""
    grouped = defaultdict(list)
    counts = set()
    for record in records:
        _validate_record(record)
        if record.status == "ok" and record.operation == "lookup_miss" and record.algorithm in ("bloom", "cuckoo"):
            grouped[record.algorithm, record.n].append(record.false_positives / record.operations * 100)
            counts.add(record.operations)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    for algorithm in ("bloom", "cuckoo"):
        xs = sorted(n for a, n in grouped if a == algorithm)
        if xs:
            quantiles = np.array([np.quantile(grouped[algorithm, n], (0.25, 0.5, 0.75)) for n in xs])
            _line(ax, algorithm, xs, quantiles[:, 1], quantiles[:, 0], quantiles[:, 2])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("False positives / known-absent queries (%)")
    ax.set_title("Observed filter false-positive rate")
    _finish(fig, [ax], records, f"Median and IQR across successful repetitions; absent queries per repetition: {', '.join(map(str, sorted(counts))) or 'none'}.\nZero observed false positives does not establish a zero theoretical rate.")
    return fig
