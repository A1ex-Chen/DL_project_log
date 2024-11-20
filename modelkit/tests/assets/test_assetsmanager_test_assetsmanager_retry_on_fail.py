def test_assetsmanager_retry_on_fail(base_dir, working_dir):
    bucket_path = os.path.join(base_dir, 'local_driver', 'bucket')
    os.makedirs(bucket_path)
    mng = AssetsManager(assets_dir=working_dir, storage_provider=
        StorageProvider(provider='local', bucket=bucket_path))
    data_path = os.path.join(test_path, 'testdata', 'some_data.json')
    mng.storage_provider.push(data_path, 'category-test/some-data.ext', '1.0')
    asset_info = mng.fetch_asset('category-test/some-data.ext:1.0',
        return_info=True)
    assert not asset_info['from_cache']
    assert os.path.exists(_success_file_path(asset_info['path']))
    os.unlink(_success_file_path(asset_info['path']))
    asset_info = mng.fetch_asset('category-test/some-data.ext:1.0',
        return_info=True)
    assert not asset_info['from_cache']
    data_path = os.path.join(test_path, 'testdata')
    mng.storage_provider.push(data_path, 'category-test/some-data-dir', '1.0')
    asset_info = mng.fetch_asset('category-test/some-data-dir:1.0',
        return_info=True)
    assert not asset_info['from_cache']
    assert os.path.exists(_success_file_path(asset_info['path']))
    os.unlink(_success_file_path(asset_info['path']))
    asset_info = mng.fetch_asset('category-test/some-data-dir:1.0',
        return_info=True)
    assert not asset_info['from_cache']
