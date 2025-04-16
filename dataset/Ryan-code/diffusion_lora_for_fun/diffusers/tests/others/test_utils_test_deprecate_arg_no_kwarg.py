def test_deprecate_arg_no_kwarg(self):
    with self.assertWarns(FutureWarning) as warning:
        deprecate(('deprecated_arg', self.higher_version, 'message'))
    assert str(warning.warning
        ) == f'`deprecated_arg` is deprecated and will be removed in version {self.higher_version}. message'
