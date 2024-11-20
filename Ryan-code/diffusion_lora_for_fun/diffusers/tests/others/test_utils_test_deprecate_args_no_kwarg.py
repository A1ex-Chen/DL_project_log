def test_deprecate_args_no_kwarg(self):
    with self.assertWarns(FutureWarning) as warning:
        deprecate(('deprecated_arg_1', self.higher_version, 'Hey'), (
            'deprecated_arg_2', self.higher_version, 'Hey'))
    assert str(warning.warnings[0].message
        ) == f'`deprecated_arg_1` is deprecated and will be removed in version {self.higher_version}. Hey'
    assert str(warning.warnings[1].message
        ) == f'`deprecated_arg_2` is deprecated and will be removed in version {self.higher_version}. Hey'
