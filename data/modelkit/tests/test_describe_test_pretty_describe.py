@pytest.mark.parametrize('value', ['something', 1, 2, None, {'x': 1}, [1, 2,
    3], [1, 2, 3, [4]], object(), SomePydanticModel(), int, SomeObject(),
    float, Any, lambda x: 1, b'ok', (x for x in range(10))])
def test_pretty_describe(value):
    describe(value)
