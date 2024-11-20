@pytest.mark.parametrize(*test_versioning.INIT_VERSIONING_PARAMETRIZE)
def test_fetch_asset_version_no_storage_provider(version_asset_name,
    version, versioning):
    manager = AssetsManager(assets_dir=os.path.join(TEST_DIR, 'testdata',
        'test-bucket', 'assets-prefix'))
    asset_name = os.path.join('category', version_asset_name)
    spec = AssetSpec(name=asset_name, version=version, versioning=versioning)
    asset_dict = manager._fetch_asset_version(spec=spec, _force_download=False)
    assert asset_dict == {'from_cache': True, 'version': version, 'path':
        os.path.join(manager.assets_dir, asset_name, version)}
    with pytest.raises(errors.StorageDriverError):
        manager._fetch_asset_version(spec=spec, _force_download=True)
    spec.name = os.path.join('not-existing-asset', version_asset_name)
    with pytest.raises(errors.LocalAssetDoesNotExistError):
        manager._fetch_asset_version(spec=spec, _force_download=False)
