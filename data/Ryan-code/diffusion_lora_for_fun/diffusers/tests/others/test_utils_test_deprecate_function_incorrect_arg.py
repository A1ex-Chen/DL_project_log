def test_deprecate_function_incorrect_arg(self):
    kwargs = {'deprecated_arg': 4}
    with self.assertRaises(TypeError) as error:
        deprecate(('wrong_arg', self.higher_version, 'message'), take_from=
            kwargs)
    assert 'test_deprecate_function_incorrect_arg in' in str(error.exception)
    assert 'line' in str(error.exception)
    assert 'got an unexpected keyword argument `deprecated_arg`' in str(error
        .exception)
