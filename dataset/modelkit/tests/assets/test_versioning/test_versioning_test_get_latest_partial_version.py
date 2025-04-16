def test_get_latest_partial_version():
    assert AssetsVersioningSystem.get_latest_partial_version('any_version',
        ['any', 'version', 'list']) == 'any'
