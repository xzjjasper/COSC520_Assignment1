import csv
from dataclasses import replace
import json
import os

import pytest
import cosc520_assignment1.benchmark as module
from cosc520_assignment1 import BloomFilter, CuckooInsertionError
from cosc520_assignment1.datasets import generate_dataset, save_dataset


def small_config(**kwargs):
    return replace(module.BenchmarkConfig(sizes=(128,), repeats=1, workers=1,
                   queries_per_kind=17, warmup_queries=3, query_batch_size=4), **kwargs)


def test_all_algorithms_measure_true_categories_and_storage():
    records = module.run_benchmarks(small_config())
    assert len(records) == 15
    assert all(record.status == 'ok' for record in records)
    assert all(record.structure_bytes > 0 for record in records)
    assert all(record.elapsed_seconds >= 0 for record in records)
    assert all(record.operations == 17 for record in records if record.operation != 'build')
    assert all(record.worker_pid == os.getpid() for record in records)
    assert all(record.false_negatives == 0 for record in records if record.operation != 'build')


def test_false_positives_stay_in_absent_category(monkeypatch):
    checker = BloomFilter(128)
    monkeypatch.setattr(checker, 'contains', lambda key: True)
    monkeypatch.setattr(module, 'build_checker', lambda *args: checker)
    records = module.run_benchmarks(small_config(algorithms=('bloom',)))
    hit = next(r for r in records if r.operation == 'lookup_hit')
    miss = next(r for r in records if r.operation == 'lookup_miss')
    assert hit.false_positives == 0 and hit.false_negatives == 0
    assert miss.false_positives == 17 and miss.operations == 17 and miss.status == 'ok'


def test_failure_and_memory_skip_never_become_zero_times(monkeypatch):
    def fail(*args):
        raise CuckooInsertionError('forced failure')
    monkeypatch.setattr(module, 'build_checker', fail)
    for config, expected in ((small_config(algorithms=('cuckoo',)), 'error'),
                             (small_config(algorithms=('linear',), memory_budget_bytes=1), 'skipped')):
        records = module.run_benchmarks(config)
        assert len(records) == 1
        assert records[0].status == expected and records[0].error
        assert records[0].elapsed_seconds is None and records[0].structure_bytes is None


def test_queries_are_shuffled_without_changing_labels():
    dataset = generate_dataset(100, 17)
    config = small_config()
    first = module._query_schedule(dataset, config, 0)
    assert first == module._query_schedule(dataset, config, 0)
    for kind, expected in (('lookup_hit', dataset.hits), ('lookup_miss', dataset.misses)):
        actual = [key for category, keys in first if category == kind for key in keys]
        assert sorted(actual) == sorted(expected)
    assert {kind for kind, _ in first} == {'lookup_hit', 'lookup_miss'}


def test_saved_provenance_and_csv_round_trip(tmp_path):
    source, output = tmp_path / 'data', tmp_path / 'run'
    save_dataset(generate_dataset(1000, 20), source)
    records = module.run_benchmarks(small_config(), data_dir=source, output_dir=output)
    assert module.load_results(output / 'raw.csv') == records
    metadata = json.loads((output / 'environment.json').read_text())
    assert metadata['worker_pids'] == [os.getpid()]
    assert metadata['config']['workers'] == 1 and metadata['config']['login_length'] == 7
    assert not (output / '.incomplete').exists()
    assert len((output / 'queries-128' / 'hits.txt').read_text().splitlines()) == 17
    with pytest.raises(FileExistsError):
        module.save_results(records, output / 'raw.csv')
    with (output / 'raw.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        names = [name for name in reader.fieldnames if name != 'worker_pid']
    with (tmp_path / 'legacy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows({name: row[name] for name in names} for row in rows)
    assert all(r.worker_pid is None for r in module.load_results(tmp_path / 'legacy.csv'))


def test_spawn_workers_complete_jobs_in_two_processes():
    config = small_config(sizes=(20_000,), algorithms=('linear', 'binary'), workers=2,
                          repeats=2, queries_per_kind=500)
    records = module.run_benchmarks(config)
    assert len(records) == 12 and all(r.status == 'ok' for r in records)
    pids = {r.worker_pid for r in records}
    assert len(pids) == 2 and os.getpid() not in pids
    assert all(r.false_negatives == 0 for r in records if r.operation != 'build')
    assert len({(r.algorithm, r.n, r.repeat, r.operation) for r in records}) == 12


@pytest.mark.parametrize('workers', [0, -1])
def test_invalid_worker_counts(workers):
    with pytest.raises(ValueError):
        small_config(workers=workers).validate()


def test_shared_memory_budget_allows_large_job_but_limits_total(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from types import SimpleNamespace
    import time

    costs = {'linear': 80, 'binary': 40, 'bloom': 20}
    monkeypatch.setattr(module.psutil, 'virtual_memory', lambda: SimpleNamespace(available=150))
    monkeypatch.setattr(module, 'estimate_build_bytes', lambda a, n, c: costs[a])
    lock = Lock()
    active = 0
    peak = 0

    def measure(dataset, config, algorithm, repeat, budget):
        nonlocal active, peak
        assert budget == 100
        with lock:
            active += costs[algorithm]
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= costs[algorithm]
        return [module.BenchmarkRecord(algorithm, len(dataset.logins), repeat, 'build',
                len(dataset.logins), 0.02, None, None, 100)]

    monkeypatch.setattr(module, '_benchmark_one', measure)
    config = small_config(workers=2, algorithms=tuple(costs), memory_budget_bytes=100)
    checkpoints = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = module.benchmark_dataset(generate_dataset(128, 17), config, executor=pool,
                                       checkpoint=lambda records: checkpoints.append(len(records)))
    assert len(rows) == 3 and all(r.status == 'ok' for r in rows)
    assert 80 <= peak <= 105
    assert checkpoints == [1, 2, 3]


def test_large_cuckoo_size_measurement_uses_bounded_auxiliary_memory():
    import tracemalloc
    from cosc520_assignment1 import CuckooFilter
    checker = CuckooFilter(bucket_count=16384)
    tracemalloc.start()
    try:
        size = module.measure_structure_bytes(checker)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert size > 16384 * 4
    assert peak < 200_000  # No set containing every independent bucket's identity.


def test_long_run_writes_completed_jobs_before_finishing(tmp_path, monkeypatch):
    original = module._benchmark_one
    seen_checkpoints = []
    output = tmp_path / 'run'

    def inspect_checkpoint(*args):
        checkpoint = output / 'partial.csv'
        seen_checkpoints.append(len(module.load_results(checkpoint)) if checkpoint.exists() else 0)
        return original(*args)

    monkeypatch.setattr(module, '_benchmark_one', inspect_checkpoint)
    rows = module.run_benchmarks(small_config(), output_dir=output)
    assert seen_checkpoints == [0, 3, 6, 9, 12]
    assert len(rows) == 15 and not (output / 'partial.csv').exists()
    assert not (output / '.incomplete').exists()
