def test_asset_spec_get_local_versions():
    spec = AssetSpec(name='name', versioning='simple_date')
    assert spec.get_local_versions('not_a_dir') == []
    asset_dir = ['testdata', 'test-bucket', 'assets-prefix', 'category',
        'simple_date_asset']
    local_path = os.path.join(tests.TEST_DIR, *asset_dir)
    assert spec.get_local_versions(local_path) == ['2021-11-15T17-31-06Z',
        '2021-11-14T18-00-00Z']
