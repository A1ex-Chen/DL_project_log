@pytest.mark.parametrize(*test_versioning.INIT_VERSIONING_PARAMETRIZE)
def test_fetch_asset_version_with_storage_provider(version_asset_name,
    version, versioning, working_dir):
    manager = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(provider='local', bucket=os.path.join(TEST_DIR,
        'testdata', 'test-bucket'), prefix='assets-prefix'))
    asset_name = os.path.join('category', version_asset_name)
    spec = AssetSpec(name=asset_name, version=version, versioning=versioning)
    asset_dict = manager._fetch_asset_version(spec=spec, _force_download=False)
    del asset_dict['meta']
    assert asset_dict == {'from_cache': False, 'version': version, 'path':
        os.path.join(working_dir, asset_name, version)}
    asset_dict = manager._fetch_asset_version(spec=spec, _force_download=False)
    assert asset_dict == {'from_cache': True, 'version': version, 'path':
        os.path.join(working_dir, asset_name, version)}
    asset_dict = manager._fetch_asset_version(spec=spec, _force_download=True)
    del asset_dict['meta']
    assert asset_dict == {'from_cache': False, 'version': version, 'path':
        os.path.join(working_dir, asset_name, version)}
    os.remove(os.path.join(working_dir, asset_name, version))
    asset_dict = manager._fetch_asset_version(spec=spec, _force_download=False)
    del asset_dict['meta']
    assert asset_dict == {'from_cache': False, 'version': version, 'path':
        os.path.join(working_dir, asset_name, version)}
