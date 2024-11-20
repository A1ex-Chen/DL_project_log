def test_asset_spec_get_local_versions():
    spec = AssetSpec(name='name', versioning='major_minor')
    assert spec.get_local_versions('not_a_dir') == []
    asset_dir = ['testdata', 'test-bucket', 'assets-prefix', 'category',
        'asset']
    local_path = os.path.join(tests.TEST_DIR, *asset_dir)
    assert spec.get_local_versions(local_path) == ['1.0', '0.1', '0.0']
