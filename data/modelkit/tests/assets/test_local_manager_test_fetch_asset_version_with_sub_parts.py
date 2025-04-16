@pytest.mark.parametrize(*test_versioning.INIT_VERSIONING_PARAMETRIZE)
def test_fetch_asset_version_with_sub_parts(version_asset_name, version,
    versioning, working_dir):
    manager = AssetsManager(assets_dir=os.path.join(TEST_DIR, 'testdata',
        'test-bucket', 'assets-prefix'))
    asset_name = os.path.join('category', version_asset_name)
    sub_part = 'sub_part'
    spec = AssetSpec(name=asset_name, version=version, sub_part=sub_part,
        versioning=versioning)
    asset_dict = manager._fetch_asset_version(spec=spec, _force_download=False)
    assert asset_dict == {'from_cache': True, 'version': version, 'path':
        os.path.join(manager.assets_dir, asset_name, version, sub_part)}
