import pytest
from cosc520_assignment1.bloom_filter import BloomFilter, bloom_parameters


def test_normal_parameters_and_hash_bounds():
    bits, hashes = bloom_parameters(1000, 0.01)
    assert bits > 1000 and hashes > 0
    filter_ = BloomFilter(1000)
    indexes = list(filter_._hash_indexes('key'))
    assert len(indexes) == hashes
    assert all(0 <= i < bits for i in indexes)


def test_packed_bit_boundaries():
    filter_ = BloomFilter(100)
    indexes = [0, 7, 8, filter_._bit_count - 1]
    for index in indexes:
        filter_._set_bit(index)
    assert all(filter_._get_bit(index) == (index in indexes) for index in range(filter_._bit_count))
    with pytest.raises(IndexError):
        filter_._get_bit(filter_._bit_count)


def test_no_false_negatives_for_normal_configuration():
    keys = [f'key-{i}' for i in range(1000)]
    filter_ = BloomFilter(1000)
    assert not filter_.contains('anything')
    for key in keys:
        filter_.add(key)
    assert all(filter_.contains(key) for key in keys)
