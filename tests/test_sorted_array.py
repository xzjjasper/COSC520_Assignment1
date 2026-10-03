from bisect import bisect_left
from cosc520_assignment1.sorted_array import SortedArray


def test_sort_deduplicates_without_mutating_input():
    keys = ['z', 'a', 'a', 'm']
    checker = SortedArray(keys)
    assert checker._items == ['a', 'm', 'z']
    assert keys == ['z', 'a', 'a', 'm']


def test_lower_bound_boundaries():
    for values in ([], ['c'], ['b', 'd', 'd', 'f']):
        checker = SortedArray(values)
        for key in ('', 'a', 'b', 'c', 'd', 'e', 'f', 'z'):
            assert checker._lower_bound(key) == bisect_left(sorted(set(values)), key)


def test_add_preserves_order():
    checker = SortedArray(['b', 'd'])
    for key in ('a', 'c', 'e', 'b'):
        checker.add(key)
    assert checker._items == list('abcde')
