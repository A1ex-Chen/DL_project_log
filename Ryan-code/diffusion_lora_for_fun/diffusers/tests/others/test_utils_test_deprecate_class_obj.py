def test_deprecate_class_obj(self):


    class Args:
        arg = 5
    with self.assertWarns(FutureWarning) as warning:
        arg = deprecate(('arg', self.higher_version, 'message'), take_from=
            Args())
    assert arg == 5
    assert str(warning.warning
        ) == f'The `arg` attribute is deprecated and will be removed in version {self.higher_version}. message'
