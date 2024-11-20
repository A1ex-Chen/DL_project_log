def test_fetch_local_version():
    asset_name = os.path.join('category', 'asset')
    local_name = os.path.join(TEST_DIR, 'testdata', 'test-bucket',
        'assets-prefix', asset_name)
    assert _fetch_local_version('', local_name) == {'path': local_name}
    assert _fetch_local_version('README.md', '') == {'path': os.path.join(
        os.getcwd(), 'README.md')}
    asset_name = os.path.join(os.getcwd(), 'README.md')
    assert _fetch_local_version(asset_name, '') == {'path': asset_name}
    with pytest.raises(errors.AssetDoesNotExistError):
        _fetch_local_version('asset/not/exists', '')
