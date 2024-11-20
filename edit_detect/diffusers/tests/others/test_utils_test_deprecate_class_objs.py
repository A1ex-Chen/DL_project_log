def test_deprecate_class_objs(self):


    class Args:
        arg = 5
        foo = 7
    with self.assertWarns(FutureWarning) as warning:
        arg_1, arg_2 = deprecate(('arg', self.higher_version, 'message'), (
            'foo', self.higher_version, 'message'), ('does not exist', self
            .higher_version, 'message'), take_from=Args())
    assert arg_1 == 5
    assert arg_2 == 7
    assert str(warning.warning
        ) == f'The `arg` attribute is deprecated and will be removed in version {self.higher_version}. message'
    assert str(warning.warnings[0].message
        ) == f'The `arg` attribute is deprecated and will be removed in version {self.higher_version}. message'
    assert str(warning.warnings[1].message
        ) == f'The `foo` attribute is deprecated and will be removed in version {self.higher_version}. message'
