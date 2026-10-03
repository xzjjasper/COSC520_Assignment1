import random
import pytest
from cosc520_assignment1 import CuckooFilter, CuckooInsertionError
from cosc520_assignment1.datasets import generate_dataset


@pytest.mark.parametrize('bucket_count', [2, 16, 1024])
def test_alternate_bucket_involution(bucket_count):
    filter_ = CuckooFilter(bucket_count=bucket_count)
    for index in range(bucket_count):
        for fingerprint in (1, 2, 127, 255):
            alternate = filter_._alternate_index(index, fingerprint)
            assert 0 <= alternate < bucket_count
            assert filter_._alternate_index(alternate, fingerprint) == index


@pytest.mark.parametrize('bits', [1, 4, 8])
def test_nonzero_fingerprint_range(bits):
    filter_ = CuckooFilter(fingerprint_bits=bits)
    for key in ('', 'a', '用户名', *map(str, range(1000))):
        assert 1 <= filter_._fingerprint(key) < 2**bits


def test_no_false_negatives_and_seeded_reproducibility():
    keys = list(generate_dataset(2000, 20).logins)
    before = random.getstate()
    first = CuckooFilter(bucket_count=1024, logins=keys, seed=42)
    second = CuckooFilter(bucket_count=1024, logins=keys, seed=42)
    assert random.getstate() == before
    assert first._buckets == second._buckets
    assert all(first.contains(key) for key in keys)


def test_failure_restores_revisited_slots(monkeypatch):
    filter_ = CuckooFilter(bucket_count=2, bucket_size=2, max_kicks=31)
    filter_._buckets = [bytearray([1, 2]), bytearray([3, 4])]
    monkeypatch.setattr(filter_, '_alternate_index', lambda index, fp: 1 - index)
    before = [bytes(bucket) for bucket in filter_._buckets]
    with pytest.raises(CuckooInsertionError):
        filter_._relocate(99, 0)
    assert [bytes(bucket) for bucket in filter_._buckets] == before


def test_random_nonzero_slot_and_last_kick_can_succeed(monkeypatch):
    filter_ = CuckooFilter(bucket_count=2, bucket_size=2, max_kicks=1, seed=0)
    filter_._buckets = [bytearray([1, 2]), bytearray([0, 0])]
    monkeypatch.setattr(filter_, '_alternate_index', lambda index, fp: 1 if fp == 2 else 0)
    filter_._relocate(9, 0)
    assert filter_._buckets == [bytearray([1, 9]), bytearray([2, 0])]


def test_failed_real_insert_preserves_all_accepted_keys():
    filter_ = CuckooFilter(bucket_count=2, bucket_size=2, max_kicks=20, seed=42)
    accepted = []
    for i in range(20):
        before = [bytes(bucket) for bucket in filter_._buckets]
        try:
            filter_.add(f'key-{i}')
        except CuckooInsertionError:
            assert [bytes(bucket) for bucket in filter_._buckets] == before
            assert all(filter_.contains(key) for key in accepted)
            return
        accepted.append(f'key-{i}')
    pytest.fail('a bounded table must eventually reject an insertion')
