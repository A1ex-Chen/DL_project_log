def test_load_model(working_dir, monkeypatch):


    class MyModel(modelkit.Model):
        CONFIGURATIONS = {'my_model': {'asset':
            'category/simple_date_asset:2021-11-14T18-00-00Z'},
            'my_last_model': {'asset': 'category/simple_date_asset'}}

        def _load(self):
            with open(self.asset_path) as f:
                self.data = json.load(f)

        def _predict(self, item, **kwargs):
            return self.data['name']
    monkeypatch.setenv('MODELKIT_ASSETS_DIR', working_dir)
    monkeypatch.setenv('MODELKIT_ASSETS_VERSIONING_SYSTEM', 'simple_date')
    monkeypatch.setenv('MODELKIT_STORAGE_PROVIDER', 'local')
    monkeypatch.setenv('MODELKIT_STORAGE_BUCKET', os.path.join(tests.
        TEST_DIR, 'testdata', 'test-bucket'))
    monkeypatch.setenv('MODELKIT_STORAGE_PREFIX', 'assets-prefix')
    model = modelkit.load_model('my_model', models=MyModel)
    assert model.predict({}) == 'asset-2021-11-14T18-00-00Z'
    my_last_model = modelkit.load_model('my_last_model', models=MyModel)
    assert my_last_model.predict({}) == 'asset-2021-11-15T17-31-06Z'
