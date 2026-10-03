"""Run measured experiments; algorithm failures are retained as diagnostics."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

from cosc520_assignment1.benchmark import ALGORITHMS, DEFAULT_SIZES, BenchmarkConfig, run_benchmarks


def main() -> None:
    """Parse experiment settings, save real raw results and print their status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--workers", "--threads", dest="workers", type=int, default=2, help="number of worker processes; 1 runs serially")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queries", type=int, default=1000, help="queries per true membership category")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64, help="use 1 to shuffle individual hit/miss queries; timer overhead will increase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", type=int, choices=(7, 8), default=7)
    parser.add_argument("--bloom-rate", type=float, default=0.01)
    parser.add_argument("--cuckoo-bucket-size", type=int, default=4)
    parser.add_argument("--cuckoo-bits", type=int, default=8)
    parser.add_argument("--cuckoo-kicks", type=int, default=500)
    parser.add_argument("--cuckoo-load", type=float, default=0.90)
    parser.add_argument("--memory-budget-gib", type=float, default=12.0)
    parser.add_argument("--data", type=Path, default=Path("data/hundred-million"), help="existing master dataset directory")
    parser.add_argument("--output", type=Path, default=None, help="new directory; existing outputs are never overwritten")
    args = parser.parse_args()
    config = BenchmarkConfig(sizes=tuple(args.sizes), algorithms=tuple(args.algorithms),
        repeats=args.repeats, workers=args.workers, queries_per_kind=args.queries, warmup_queries=args.warmup,
        query_batch_size=args.batch_size, seed=args.seed, login_length=args.length,
        bloom_false_positive_rate=args.bloom_rate, cuckoo_bucket_size=args.cuckoo_bucket_size,
        cuckoo_fingerprint_bits=args.cuckoo_bits, cuckoo_max_kicks=args.cuckoo_kicks,
        cuckoo_target_load=args.cuckoo_load, memory_budget_bytes=int(args.memory_budget_gib * 1024**3))
    output = args.output or Path("results") / datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    if "PYTHONHASHSEED" not in os.environ:
        print("PYTHONHASHSEED is unfixed; for repeatability launch with PYTHONHASHSEED=42. Setting it inside Python is too late.")
    records = run_benchmarks(config, data_dir=args.data, output_dir=output, progress=lambda message: print(message, flush=True))
    failures = {(r.algorithm, r.n, r.repeat, r.error) for r in records if r.status != "ok"}
    print(f"Saved {len(records)} raw records to {output / 'raw.csv'}")
    for algorithm, n, repeat, error in sorted(failures):
        print(f"{algorithm} n={n:,} repetition={repeat + 1}: {error}")
    print(f"{len(failures)} unsuccessful repetitions; inspect status/error columns before interpreting results.")
    if failures:
        raise SystemExit(2)  # Results are still exported, but a partial run is not a complete success.


if __name__ == "__main__":
    main()
