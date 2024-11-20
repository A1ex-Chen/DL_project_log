def test_deprecate_function_args(self):
    kwargs = {'deprecated_arg_1': 4, 'deprecated_arg_2': 8}
    with self.assertWarns(FutureWarning) as warning:
        output_1, output_2 = deprecate(('deprecated_arg_1', self.
            higher_version, 'Hey'), ('deprecated_arg_2', self.
            higher_version, 'Hey'), take_from=kwargs)
    assert output_1 == 4
    assert output_2 == 8
    assert str(warning.warnings[0].message
        ) == f'The `deprecated_arg_1` argument is deprecated and will be removed in version {self.higher_version}. Hey'
    assert str(warning.warnings[1].message
        ) == f'The `deprecated_arg_2` argument is deprecated and will be removed in version {self.higher_version}. Hey'
