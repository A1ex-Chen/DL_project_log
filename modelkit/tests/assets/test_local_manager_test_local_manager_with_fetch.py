@pytest.mark.parametrize(*test_versioning.TWO_VERSIONING_PARAMETRIZE)
def test_local_manager_with_fetch(version_asset_name, version_1, version_2,
    versioning, working_dir, monkeypatch):
    if versioning:
        monkeypatch.setenv('MODELKIT_ASSETS_VERSIONING_SYSTEM', versioning)
    manager = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(provider='local', bucket=os.path.join(TEST_DIR,
        'testdata', 'test-bucket'), prefix='assets-prefix'))
    res = manager.fetch_asset(f'category/{version_asset_name}:{version_1}',
        return_info=True)
    assert res['path'] == os.path.join(working_dir, 'category',
        version_asset_name, version_1)
    res = manager.fetch_asset(f'category/{version_asset_name}', return_info
        =True)
    assert res['path'] == os.path.join(working_dir, 'category',
        version_asset_name, version_2)
    if versioning in ['major_minor', None]:
        res = manager.fetch_asset(f'category/{version_asset_name}:0',
            return_info=True)
        assert res['path'] == os.path.join(working_dir, 'category',
            version_asset_name, '0.1')
