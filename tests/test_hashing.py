from cosc520_assignment1.hashing import hash_string


def test_deterministic_seeded_unicode_hash():
    for value in ('', 'a', '用户名'):
        assert 0 <= hash_string(value, 42) < 2**64
        assert hash_string(value, 42) == hash_string(value, 42)
        assert hash_string(value, 42) != hash_string(value, 43)
