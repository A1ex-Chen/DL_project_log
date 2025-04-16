@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)
def test_download_assets_version(version_asset_name, version_1, version_2,
    versioning, assetsmanager_settings, monkeypatch):
    if versioning:
        monkeypatch.setenv('MODELKIT_ASSETS_VERSIONING_SYSTEM', versioning)


    class SomeModel(Asset):
        CONFIGURATIONS = {'model0': {'asset':
            f'category/{version_asset_name}:{version_1}'}}
    model_assets, assets_info = download_assets(assetsmanager_settings=
        assetsmanager_settings, models=[SomeModel])
    assert model_assets['model0'] == {
        f'category/{version_asset_name}:{version_1}'}
    assert assets_info[f'category/{version_asset_name}:{version_1}'
        ].version == version_1


    class SomeModel(Asset):
        CONFIGURATIONS = {'model0': {'asset': f'category/{version_asset_name}'}
            }
    model_assets, assets_info = download_assets(assetsmanager_settings=
        assetsmanager_settings, models=[SomeModel])
    assert model_assets['model0'] == {f'category/{version_asset_name}'}
    assert assets_info[f'category/{version_asset_name}'].version == version_2
    if versioning in (None, 'major_minor'):


        class SomeModel(Asset):
            CONFIGURATIONS = {'model0': {'asset': 'category/asset:0'}}
        model_assets, assets_info = download_assets(assetsmanager_settings=
            assetsmanager_settings, models=[SomeModel])
        assert model_assets['model0'] == {'category/asset:0'}
        assert assets_info['category/asset:0'].version == '0.1'
