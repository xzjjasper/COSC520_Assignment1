"""Export runtime, retained-storage and observed-FPR plots from real CSV data."""
import argparse
from pathlib import Path

from matplotlib import pyplot as plt
from cosc520_assignment1.benchmark import load_results
from cosc520_assignment1.plotting import plot_runtimes, plot_storage_sizes, plot_false_positive_rates


def main() -> None:
    """Read one results CSV and save PNG/PDF figures to a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory; no overwrites")
    args = parser.parse_args()
    records = load_results(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, plotter in (("runtimes", plot_runtimes), ("storage", plot_storage_sizes),
                          ("false-positive-rates", plot_false_positive_rates)):
        figure = plotter(records)
        for suffix in ("png", "pdf"):
            figure.savefig(args.output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)
    print(f"Saved three measured figures (PNG/PDF) to {args.output}")


if __name__ == "__main__":
    main()
