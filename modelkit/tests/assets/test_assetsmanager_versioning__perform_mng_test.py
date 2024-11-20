def _perform_mng_test(mng):
    data_path = os.path.join(test_path, 'testdata', 'some_data.json')
    mng.storage_provider.new(data_path, 'category-test/some-data', '0.0',
        dry_run=True)
    with pytest.raises(errors.ObjectDoesNotExistError):
        mng.fetch_asset('category-test/some-data')
    data_path = os.path.join(test_path, 'testdata', 'some_data.json')
    with pytest.raises(errors.AssetDoesNotExistError):
        mng.storage_provider.update(data_path, 'category-test/some-data',
            version='0.0')
    mng.storage_provider.new(data_path, 'category-test/some-data', '0.0')
    mng.storage_provider.get_asset_meta('category-test/some-data', '0.0')
    mng.storage_provider.update(data_path, 'category-test/some-data',
        version='0.1', dry_run=True)
    with pytest.raises(errors.ObjectDoesNotExistError):
        mng.fetch_asset('category-test/some-data:0.1')
    mng.storage_provider.update(data_path, 'category-test/some-data',
        version='0.1')
    mng.storage_provider.get_asset_meta('category-test/some-data', '0.1')
    with pytest.raises(errors.AssetAlreadyExistsError):
        mng.storage_provider.update(data_path, 'category-test/some-data',
            version='0.1')
    with pytest.raises(errors.AssetAlreadyExistsError):
        mng.storage_provider.new(data_path, 'category-test/some-data',
            version='0.0')
    mng.storage_provider.update(data_path, 'category-test/some-data',
        version='1.0')
    mng.storage_provider.get_asset_meta('category-test/some-data', '1.0')
    fetched_path = mng.fetch_asset('category-test/some-data:1.0')
    assert filecmp.cmp(fetched_path, data_path)
    fetched_path = mng.fetch_asset('category-test/some-data:1')
    assert filecmp.cmp(fetched_path, data_path)
    fetched_path = mng.fetch_asset('category-test/some-data')
    assert filecmp.cmp(fetched_path, data_path)
    fetched_asset_dict = mng.fetch_asset('category-test/some-data',
        return_info=True)
    assert fetched_asset_dict['path'], fetched_path
    assert fetched_asset_dict['from_cache'] is True
    assert fetched_asset_dict['version'] == '1.0'
    assert list(mng.storage_provider.iterate_assets()) == [(
        'category-test/some-data', ['1.0', '0.1', '0.0'])]
    mng.storage_provider.update(data_path, 'category-test/some-data',
        version='1.1')
    mng.storage_provider.get_asset_meta('category-test/some-data', '1.1')
    fetched_asset_dict = mng.fetch_asset('category-test/some-data',
        return_info=True)
    assert fetched_asset_dict['path'], fetched_path
    assert fetched_asset_dict['from_cache'] is False
    assert fetched_asset_dict['version'] == '1.1'
    assert list(mng.storage_provider.iterate_assets()) == [(
        'category-test/some-data', ['1.1', '1.0', '0.1', '0.0'])]
