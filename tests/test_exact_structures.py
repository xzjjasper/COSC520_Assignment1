import pytest
from cosc520_assignment1 import LinearSearch, SortedArray, HashTable


@pytest.mark.parametrize('checker_type', [LinearSearch, SortedArray, HashTable])
def test_constructor_membership_and_existing_key(checker_type):
    keys = ['', 'Alice', 'alice', '用户名', '0', *[f'key-{i}' for i in range(100)]]
    checker = checker_type(keys)
    for key in keys + ['ALICE', 'missing', '你好']:
        assert checker.contains(key) == (key in keys)
    checker.add('Alice')
    assert all(checker.contains(key) for key in keys)


@pytest.mark.parametrize('checker_type', [LinearSearch, SortedArray, HashTable])
def test_empty_membership(checker_type):
    assert not checker_type().contains('missing')
