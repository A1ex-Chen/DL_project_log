def _perform_mng_test_subpart(mng):
    data_path = os.path.join(TEST_DIR, 'assets', 'testdata', 'some_data_folder'
        )
    mng.storage_provider.new(data_path, 'category-test/some-data-subpart',
        '0.0')
    fetched_asset_dict = mng.fetch_asset(
        'category-test/some-data-subpart:0.0[/some_data_in_folder.json]',
        return_info=True)
    assert filecmp.cmp(fetched_asset_dict['path'], os.path.join(data_path,
        'some_data_in_folder.json'))
    fetched_asset_dict = mng.fetch_asset(
        'category-test/some-data-subpart:0.0[some_data_in_folder.json]',
        return_info=True)
    assert filecmp.cmp(fetched_asset_dict['path'], os.path.join(data_path,
        'some_data_in_folder.json'))
    fetched_asset_dict = mng.fetch_asset(
        'category-test/some-data-subpart:0.0[some_data_in_folder_2.json]',
        return_info=True)
    assert filecmp.cmp(fetched_asset_dict['path'], os.path.join(data_path,
        'some_data_in_folder_2.json'))
    data_path = os.path.join(TEST_DIR, 'assets', 'testdata')
    mng.storage_provider.new(data_path, 'category-test/some-data-subpart-2',
        '0.0')
    fetched_asset_dict = mng.fetch_asset(
        'category-test/some-data-subpart-2:0.0[some_data.json]',
        return_info=True)
    assert filecmp.cmp(fetched_asset_dict['path'], os.path.join(data_path,
        'some_data.json'))
    fetched_asset_dict = mng.fetch_asset(
        'category-test/some-data-subpart-2:0.0[some_data_folder/some_data_in_folder_2.json]'
        , return_info=True)
    assert filecmp.cmp(fetched_asset_dict['path'], os.path.join(data_path,
        'some_data_folder', 'some_data_in_folder_2.json'))
