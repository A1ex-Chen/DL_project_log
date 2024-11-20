def test_deprecate_incorrect_no_standard_warn(self):
    with self.assertWarns(FutureWarning) as warning:
        deprecate(('deprecated_arg', self.higher_version,
            'This message is better!!!'), standard_warn=False)
    assert str(warning.warning) == 'This message is better!!!'
