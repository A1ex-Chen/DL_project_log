@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)
def test_download_assets_dependencies(version_asset_name, version_1,
    version_2, versioning, assetsmanager_settings, monkeypatch):
    if versioning:
        monkeypatch.setenv('MODELKIT_ASSETS_VERSIONING_SYSTEM', versioning)


    class SomeModel(Asset):
        CONFIGURATIONS = {'model0': {'asset': f'category/{version_asset_name}'}
            }


    class SomeOtherModel(Asset):
        CONFIGURATIONS = {'model1': {'asset':
            f'category/{version_asset_name}:{version_1}',
            'model_dependencies': {'model0'}}}
    model_assets, assets_info = download_assets(assetsmanager_settings=
        assetsmanager_settings, models=[SomeModel, SomeOtherModel])
    assert model_assets['model0'] == {f'category/{version_asset_name}'}
    assert model_assets['model1'] == {
        f'category/{version_asset_name}:{version_1}',
        f'category/{version_asset_name}'}
    assert assets_info[f'category/{version_asset_name}'].version == version_2
    assert assets_info[f'category/{version_asset_name}:{version_1}'
        ].version == version_1
