def test_environment_asset_load_version(monkeypatch, assetsmanager_settings):


    class TestModel(Model):

        def _load(self):
            assert self.asset_path == 'path/to/asset'
            self.data = {'some key': 'some data'}

        def _predict(self, item, **kwargs):
            return self.data
    monkeypatch.setenv('MODELKIT_TESTS_TEST_ASSET_VERSION', 'undef')
    with pytest.raises(InvalidAssetSpecError):
        ModelLibrary(required_models=['some_asset'], configuration={
            'some_asset': ModelConfiguration(model_type=TestModel, asset=
            'tests/test_asset')}, assetsmanager_settings=assetsmanager_settings
            )
