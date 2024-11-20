def test_model_sub_class(working_dir, monkeypatch):
    monkeypatch.setenv('MODELKIT_ASSETS_DIR', working_dir)
    with open(os.path.join(working_dir, 'something.txt'), 'w') as f:
        f.write('OK')


    class BaseAsset(Asset):

        def _load(self):
            assert self.asset_path


    class DerivedAsset(BaseAsset):
        CONFIGURATIONS = {'derived': {'asset': 'something.txt'}}

        def _predict(self, item):
            return item
    lib = ModelLibrary(models=[DerivedAsset, BaseAsset])
    lib.preload()
    assert ['derived'] == list(lib.models.keys())
    lib = ModelLibrary(models=testmodels)
    lib.preload()
    assert ['derived_asset', 'derived_model'] == sorted(list(lib.models.keys())
        )
