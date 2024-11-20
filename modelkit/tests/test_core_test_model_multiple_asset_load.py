def test_model_multiple_asset_load(working_dir, monkeypatch):
    monkeypatch.setenv('MODELKIT_ASSETS_DIR', working_dir)
    with open(os.path.join(working_dir, 'something.txt'), 'w') as f:
        f.write('OK')


    class SomeModel(Model):
        CONFIGURATIONS = {'a': {'asset': 'something.txt'}}

        def _predict(self, item):
            return item


    class SomeModel2(Model):
        CONFIGURATIONS = {'b': {'asset': 'something.txt'}}

        def _predict(self, item):
            return item
    fetched = 0

    def fake_fetch_asset(asset_spec, return_info=True):
        nonlocal fetched
        fetched += 1
        return {'path': os.path.join(working_dir, 'something.txt')}
    lib = ModelLibrary(models=[SomeModel, SomeModel2], settings={
        'lazy_loading': True})
    monkeypatch.setattr(lib.assets_manager, 'fetch_asset', fake_fetch_asset)
    lib.preload()
    assert fetched == 1
