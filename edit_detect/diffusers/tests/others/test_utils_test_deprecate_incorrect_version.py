def test_deprecate_incorrect_version(self):
    kwargs = {'deprecated_arg': 4}
    with self.assertRaises(ValueError) as error:
        deprecate(('wrong_arg', self.lower_version, 'message'), take_from=
            kwargs)
    assert str(error.exception
        ) == f"The deprecation tuple ('wrong_arg', '0.0.1', 'message') should be removed since diffusers' version {__version__} is >= {self.lower_version}"
