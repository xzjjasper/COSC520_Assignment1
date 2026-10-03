import pickle
import pytest
from cosc520_assignment1.datasets import (
    ALPHABET, MASK, GeneratedLogins, _encoded_chunk, generate_dataset,
    login_for_index, save_dataset, load_dataset, subset_dataset,
)


@pytest.mark.parametrize('width', [7, 8])
def test_unique_ascii_and_true_query_labels(width):
    data = generate_dataset(50_000, 100, login_length=width)
    keys = set(data.logins)
    assert len(keys) == 50_000
    assert all(len(s) == width and set(s) <= set(ALPHABET) for s in keys)
    assert set(data.hits) <= keys
    assert set(data.misses).isdisjoint(keys)
    assert len(data.hits) == len(data.misses) == 100
    assert data.hits == generate_dataset(50_000, 100, login_length=width).hits
    assert login_for_index(123, 42, width) != login_for_index(123, 43, width)


@pytest.mark.parametrize('width', [7, 8])
@pytest.mark.parametrize('seed', [0, 42, MASK])
@pytest.mark.parametrize('region', ['start', 'billion', 'end'])
def test_vectorized_export_matches_scalar_ids(width, seed, region):
    start = {'start': 0, 'billion': 999_999_991, 'end': 62**width - 19}[region]
    raw = _encoded_chunk(start, start + 19, seed, width)
    expected = ''.join(login_for_index(i, seed, width) + '\n' for i in range(start, start + 19))
    assert raw == expected.encode('ascii')


def test_prefix_hits_and_master_misses():
    master = generate_dataset(1000, 50)
    subset = subset_dataset(master, 10, queries_per_kind=50)
    assert set(subset.hits) <= set(list(master.logins)[:10])
    assert set(subset.misses).isdisjoint(master.logins)
    assert len(subset.hits) == 50
    assert subset.logins[-1] == master.logins[9]
    assert len(pickle.dumps(GeneratedLogins(1_000_000_000, 42, 7))) < 512


@pytest.mark.parametrize('width', [7, 8])
def test_export_round_trip_and_overwrite_policy(tmp_path, width):
    data = generate_dataset(1003, 37, login_length=width)
    destination = tmp_path / 'dataset'
    save_dataset(data, destination, chunk_size=137)
    restored = load_dataset(destination, verify_checksum=True)
    assert list(restored.logins) == list(data.logins)
    assert (restored.hits, restored.misses) == (data.hits, data.misses)
    assert (destination / 'logins.txt').stat().st_size == 1003 * (width + 1)
    assert len(pickle.dumps(restored.logins)) < 1024
    assert restored.logins[17:25] == data.logins[17:25]
    with pytest.raises(FileExistsError):
        save_dataset(data, destination)


def test_corrupt_export_is_rejected(tmp_path):
    save_dataset(generate_dataset(1000, 10), tmp_path / 'dataset')
    path = tmp_path / 'dataset' / 'logins.txt'
    with path.open('r+b') as stream:
        stream.seek(100 * 8)
        stream.write(b'!')
    with pytest.raises(ValueError, match='Checksum'):
        load_dataset(path.parent, verify_checksum=True)


@pytest.mark.parametrize('width', [6, 9])
def test_unsupported_lengths(width):
    with pytest.raises(ValueError):
        generate_dataset(100, login_length=width)
