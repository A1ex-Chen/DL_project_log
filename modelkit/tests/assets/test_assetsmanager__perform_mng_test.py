def _perform_mng_test(mng):
    data_path = os.path.join(test_path, 'testdata', 'some_data.json')
    mng.storage_provider.push(data_path, 'category-test/some-data.ext', '1.0')
    meta = mng.storage_provider.get_asset_meta('category-test/some-data.ext',
        '1.0')
    assert not meta['is_directory']
    d = mng.fetch_asset('category-test/some-data.ext:1.0', return_info=True)
    fetched_path = d['path']
    assert fetched_path.endswith(os.path.join('category-test',
        'some-data.ext', '1.0'))
    assert not d['from_cache']
    assert d['meta']
    assert filecmp.cmp(fetched_path, data_path)
    data_path = os.path.join(test_path, 'testdata', 'some_data_folder')
    mng.storage_provider.push(data_path, 'category-test/some-data-2', '1.0')
    meta = mng.storage_provider.get_asset_meta('category-test/some-data-2',
        '1.0')
    assert meta['is_directory']
    d = mng.fetch_asset('category-test/some-data-2:1.0', return_info=True)
    fetched_path = d['path']
    assert fetched_path.endswith(os.path.join('category-test',
        'some-data-2', '1.0'))
    assert not d['from_cache']
    assert d['meta']
    assert not filecmp.cmpfiles(data_path, fetched_path, [
        'some_data_in_folder.json', 'some_data_in_folder_2.json'], shallow=
        False)[1]
    d = mng.fetch_asset('category-test/some-data.ext:1.0', return_info=True)
    assert d['from_cache']
    d = mng.fetch_asset('category-test/some-data-2:1.0', return_info=True)
    assert d['from_cache']
    with pytest.raises(errors.AssetAlreadyExistsError):
        mng.storage_provider.push(os.path.join(data_path,
            'some_data_in_folder.json'), 'category-test/some-data.ext', '1.0')
    d = mng.fetch_asset('category-test/some-data.ext:1.0', return_info=True)
    assert d['from_cache']
