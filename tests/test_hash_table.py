import cosc520_assignment1.hash_table as module
from cosc520_assignment1 import HashTable


def test_one_shot_iterator_is_not_consumed_before_insertion():
    keys = ['alpha', 'beta', '', '用户名']
    table = HashTable(iter(keys))
    assert table._size == len(keys)
    assert all(table.contains(key) for key in keys)
    assert not table.contains('absent')


def test_sized_input_is_iterated_only_once():
    class SizedOnePass:
        def __init__(self):
            self.iterations = 0
        def __len__(self):
            return 3
        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1
            yield from ('a', 'b', 'c')
    source = SizedOnePass()
    table = HashTable(source)
    assert source.iterations == 1
    assert all(table.contains(key) for key in ('a', 'b', 'c'))


def test_forced_collisions_and_duplicate(monkeypatch):
    monkeypatch.setattr(module, 'hash', lambda _: 0, raising=False)
    table = HashTable(['a', 'b', 'c', 'd', 'e'])
    assert all(table.contains(key) for key in 'abcde')
    assert not table.contains('missing')
    table.add('c')
    assert table._size == 5


def test_initial_empty_and_small_tables():
    for keys in ([], ['a'], ['a', 'b']):
        table = HashTable(iter(keys))
        assert all(table.contains(key) for key in keys)
        assert not table.contains('missing')
