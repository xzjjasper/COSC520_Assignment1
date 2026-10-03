"""Export the full insertion dataset without creating a billion-element RAM list."""
import argparse
from pathlib import Path
import time
from cosc520_assignment1.datasets import generate_dataset, save_dataset, load_dataset


def main() -> None:
    """Parse generation options and print bounded progress; never overwrite data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=100_000_000)
    parser.add_argument('--queries', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--length', type=int, choices=(7, 8), default=7)
    parser.add_argument('--chunk-size', type=int, default=1_000_000)
    parser.add_argument('--output', type=Path, default=Path('data/hundred-million'))
    args = parser.parse_args()
    start = time.monotonic()
    last = start

    def progress(done: int, total: int) -> None:
        """Print at most one progress report per ten seconds, plus completion."""
        nonlocal last
        now = time.monotonic()
        if now - last >= 10 or done == total:
            elapsed = now - start
            rate = done / max(elapsed, 1e-9)
            print(f'{done:,}/{total:,} ({done / total:.1%}); {rate:,.0f} rows/s; '
                  f'elapsed {elapsed:.0f}s; remaining {(total-done)/rate:.0f}s', flush=True)
            last = now

    data = generate_dataset(args.count, args.queries, seed=args.seed, login_length=args.length)
    manifest = save_dataset(data, args.output, chunk_size=args.chunk_size, progress=progress)
    loaded = load_dataset(args.output)
    print(f'Completed {len(loaded.logins):,} insertion records; manifest: {manifest}', flush=True)


if __name__ == '__main__':
    main()
