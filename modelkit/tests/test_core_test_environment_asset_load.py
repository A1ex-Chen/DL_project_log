def test_environment_asset_load(monkeypatch, assetsmanager_settings):


    class TestModel(Model):

        def _load(self):
            assert self.asset_path == 'path/to/asset'
            self.data = {'some key': 'some data'}

        def _predict(self, item, **kwargs):
            return self.data
    monkeypatch.setenv('MODELKIT_TESTS_TEST_ASSET_FILE', 'path/to/asset')
    model_library = ModelLibrary(required_models=['some_asset'],
        configuration={'some_asset': ModelConfiguration(model_type=
        TestModel, asset='tests/test_asset')}, assetsmanager_settings=
        assetsmanager_settings)
    model = model_library.get('some_asset')
    predicted = model({})
    assert predicted == {'some key': 'some data'}
