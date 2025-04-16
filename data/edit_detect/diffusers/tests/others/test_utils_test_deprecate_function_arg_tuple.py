def test_deprecate_function_arg_tuple(self):
    kwargs = {'deprecated_arg': 4}
    with self.assertWarns(FutureWarning) as warning:
        output = deprecate(('deprecated_arg', self.higher_version,
            'message'), take_from=kwargs)
    assert output == 4
    assert str(warning.warning
        ) == f'The `deprecated_arg` argument is deprecated and will be removed in version {self.higher_version}. message'
