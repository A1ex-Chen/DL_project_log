@pytest.mark.parametrize('v00, v01, v11, v10, versioning', [('0.0', '0.1',
    '1.0', '1.1', None), ('0.0', '0.1', '1.0', '1.1', 'major_minor'), (
    '0000-00-00T00-00-00Z', '0000-00-00T01-00-00Z', '0000-00-00T10-00-00Z',
    '0000-00-00T11-00-00Z', 'simple_date')])
def test_local_manager_with_versions(v00, v01, v11, v10, versioning,
    working_dir, monkeypatch):
    if versioning:
        monkeypatch.setenv('MODELKIT_ASSETS_VERSIONING_SYSTEM', versioning)
    os.makedirs(os.path.join(working_dir, 'something', v00))
    open(os.path.join(working_dir, 'something', v00, '.SUCCESS'), 'w').close()
    os.makedirs(os.path.join(working_dir, 'something', v01))
    open(os.path.join(working_dir, 'something', v01, '.SUCCESS'), 'w').close()
    os.makedirs(os.path.join(working_dir, 'something', v11, 'subpart'))
    with open(os.path.join(working_dir, 'something', v11, 'subpart',
        'deep.txt'), 'w') as f:
        f.write('OK')
    open(os.path.join(working_dir, 'something', v11, '.SUCCESS'), 'w').close()
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset(f'something:{v11}[subpart/deep.txt]',
        return_info=True)
    assert res['path'] == os.path.join(working_dir, 'something', v11,
        'subpart', 'deep.txt')
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset(f'something/{v11}/subpart/deep.txt',
        return_info=True)
    assert res['path'] == os.path.join(working_dir, 'something', v11,
        'subpart', 'deep.txt')
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset(f'something:{v00}', return_info=True)
    assert res['path'] == os.path.join(working_dir, 'something', v00)
    manager = AssetsManager(assets_dir=working_dir)
    res = manager.fetch_asset('something', return_info=True)
    assert res['path'] == os.path.join(working_dir, 'something', v11)
    if versioning in (None, 'major_minor'):
        manager = AssetsManager(assets_dir=working_dir)
        res = manager.fetch_asset('something:0', return_info=True)
        assert res['path'] == os.path.join(working_dir, 'something', v01)
    try:
        manager = AssetsManager()
        local_dir = os.path.join('tmp-local-asset', v10, 'subpart')
        os.makedirs(local_dir)
        open(os.path.join('tmp-local-asset', v10, '.SUCCESS'), 'w').close()
        shutil.copy('README.md', local_dir)
        res = manager.fetch_asset(f'tmp-local-asset:{v10}[subpart/README.md]',
            return_info=True)
        assert res['path'] == os.path.abspath(os.path.join(local_dir,
            'README.md'))
        res = manager.fetch_asset('tmp-local-asset', return_info=True)
        assert res['path'] == os.path.abspath(os.path.join(local_dir, '..'))
        abs_path_to_readme = os.path.join(os.path.abspath(local_dir),
            'README.md')
        res = manager.fetch_asset(abs_path_to_readme, return_info=True)
        assert res['path'] == abs_path_to_readme
    finally:
        shutil.rmtree('tmp-local-asset')
