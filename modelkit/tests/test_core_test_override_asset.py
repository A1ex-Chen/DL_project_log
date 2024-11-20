def test_override_asset():


    class TestModel(Model):

        def _load(self):
            pass

        def _predict(self, item, **kwargs):
            return self.asset_path


    class TestDepModel(Model):

        def _predict(self, item, **kwargs):
            return 'dep' + self.asset_path
    config = {'some_asset': ModelConfiguration(model_type=TestModel, asset=
        'asset/that/does/not/exist', model_dependencies={'dep_model'}),
        'dep_model': ModelConfiguration(model_type=TestDepModel)}
    with pytest.raises(AssetDoesNotExistError):
        model_library = ModelLibrary(required_models=['some_asset'],
            configuration=config)
    model_library = ModelLibrary(required_models={'some_asset': {
        'asset_path': '/the/path'}}, configuration=config)
    model = model_library.get('some_asset')
    assert '/the/path' == model({})
    model = model_library.get('dep_model')
    assert 'dep' == model({})
    config['dep_model'] = ModelConfiguration(model_type=TestDepModel, asset
        ='cat/someasset')
    model_library = ModelLibrary(required_models={'some_asset': {
        'asset_path': '/the/path'}, 'dep_model': {'asset_path':
        '/the/dep/path'}}, configuration=config)
    model = model_library.get('dep_model')
    assert 'dep/the/dep/path' == model({})
