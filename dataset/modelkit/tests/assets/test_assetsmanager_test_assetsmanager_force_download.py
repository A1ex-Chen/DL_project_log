def test_assetsmanager_force_download(monkeypatch, base_dir, working_dir):
    bucket_path = os.path.join(base_dir, 'local_driver', 'bucket')
    os.makedirs(bucket_path)
    mng = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(provider='local', bucket=bucket_path))
    data_path = os.path.join(test_path, 'testdata', 'some_data.json')
    mng.storage_provider.push(data_path, 'category-test/some-data.ext', '1.0')
    asset_info = mng.fetch_asset('category-test/some-data.ext:1.0',
        return_info=True)
    assert not asset_info['from_cache']
    asset_info_re = mng.fetch_asset('category-test/some-data.ext:1.0',
        return_info=True)
    assert asset_info_re['from_cache']
    mng_force = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(provider='local', bucket=bucket_path,
        force_download=True))
    asset_info_force = mng_force.fetch_asset('category-test/some-data.ext:1.0',
        return_info=True)
    assert not asset_info_force['from_cache']
    monkeypatch.setenv('MODELKIT_STORAGE_FORCE_DOWNLOAD', 'True')
    mng_force = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(provider='local', bucket=bucket_path))
    asset_info_force_env = mng_force.fetch_asset(
        'category-test/some-data.ext:1.0', return_info=True)
    assert not asset_info_force_env['from_cache']
