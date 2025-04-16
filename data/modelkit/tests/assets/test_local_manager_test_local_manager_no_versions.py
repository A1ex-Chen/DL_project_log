def test_local_manager_no_versions(working_dir):
    os.makedirs(os.path.join(working_dir, 'something', 'else'))
    with open(os.path.join(working_dir, 'something', 'else', 'deep.txt'), 'w'
        ) as f:
        f.write('OK')
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset('something/else/deep.txt', return_info=True)
    assert res['path'] == os.path.join(working_dir, 'something', 'else',
        'deep.txt')
    manager = AssetsManager()
    res = manager.fetch_asset('README.md', return_info=True)
    assert res['path'] == os.path.join(os.getcwd(), 'README.md')
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset('README.md', return_info=True)
    assert res['path'] == os.path.join(os.getcwd(), 'README.md')
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset(os.path.join(os.getcwd(), 'README.md'),
        return_info=True)
    assert res['path'] == os.path.join(os.getcwd(), 'README.md')
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset('something', return_info=True)
    assert res['path'] == os.path.join(working_dir, 'something')
    with open(os.path.join(working_dir, 'something.txt'), 'w') as f:
        f.write('OK')
    res = manager.fetch_asset('something.txt', return_info=True)
    assert res['path'] == os.path.join(working_dir, 'something.txt')
    with pytest.raises(errors.LocalAssetDoesNotExistError):
        res = manager.fetch_asset('something.txt:0.1', return_info=True)
    with pytest.raises(errors.LocalAssetDoesNotExistError):
        res = manager.fetch_asset('something.txt:0', return_info=True)
    with pytest.raises(errors.AssetDoesNotExistError):
        res = manager.fetch_asset('doesnotexist.txt', return_info=True)
