def test_override_assets_dir(assetsmanager_settings):


    class TestModel(Model):

        def _predict(self, item, **kwargs):
            return self.asset_path
    model_library = ModelLibrary(required_models=['my_model',
        'my_override_model'], configuration={'my_model': ModelConfiguration
        (model_type=TestModel, asset='category/asset'), 'my_override_model':
        ModelConfiguration(model_type=TestModel, asset=
        'category/override-asset')}, assetsmanager_settings=
        assetsmanager_settings)
    prediction = model_library.get('my_model').predict({})
    assert prediction.endswith(os.path.join('category', 'asset', '1.0'))
    prediction = model_library.get('my_override_model').predict({})
    assert prediction.endswith(os.path.join('category', 'override-asset',
        '0.0'))
    model_library_override = ModelLibrary(required_models=['my_model',
        'my_override_model'], configuration={'my_model': ModelConfiguration
        (model_type=TestModel, asset='category/asset'), 'my_override_model':
        ModelConfiguration(model_type=TestModel, asset=
        'category/override-asset')}, settings={'override_assets_dir': os.
        path.join(TEST_DIR, 'testdata', 'override-assets-dir'),
        'lazy_loading': True}, assetsmanager_settings=assetsmanager_settings)
    prediction = model_library_override.get('my_model').predict({})
    assert prediction.endswith(os.path.join('category', 'asset', '1.0'))
    prediction = model_library_override.get('my_override_model').predict({})
    assert prediction.endswith(os.path.join('category', 'override-asset',
        '0.0'))
