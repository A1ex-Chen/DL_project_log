def test_deep_format_floats():
    assert deep_format_floats(1) == 1
    assert deep_format_floats('a') == 'a'
    assert deep_format_floats(1.2) == '1.20000'
    assert deep_format_floats({'a': [1.2, 1, 2, 3]}) == {'a': ['1.20000', 1,
        2, 3]}
    assert deep_format_floats({'a': [1.2345, 3]}, depth=2) == {'a': ['1.23', 3]
        }
    assert deep_format_floats({'a': 1.2345}, depth=2) == {'a': '1.23'}
    assert deep_format_floats({'a': [1.2345, {'b': 1.2345}]}, depth=2) == {'a':
        ['1.23', {'b': '1.23'}]}
