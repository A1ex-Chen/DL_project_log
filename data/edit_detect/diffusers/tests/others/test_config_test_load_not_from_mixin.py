def test_load_not_from_mixin(self):
    with self.assertRaises(ValueError):
        ConfigMixin.load_config('dummy_path')
