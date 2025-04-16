def test_deprecate_stacklevel(self):
    with self.assertWarns(FutureWarning) as warning:
        deprecate(('deprecated_arg', self.higher_version,
            'This message is better!!!'), standard_warn=False)
    assert str(warning.warning) == 'This message is better!!!'
    assert 'diffusers/tests/others/test_utils.py' in warning.filename
