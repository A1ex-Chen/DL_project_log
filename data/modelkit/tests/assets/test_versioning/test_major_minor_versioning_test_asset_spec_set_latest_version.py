def test_asset_spec_set_latest_version():
    spec = AssetSpec(name='a', versioning='major_minor')
    spec.set_latest_version(['3', '2.1', '1.3'])
    assert spec.version == '3'
    spec = AssetSpec(name='a', version='2', versioning='major_minor')
    spec.set_latest_version(['3', '2.1', '2.0', '1.3'])
    assert spec.version == '2.1'
    spec = AssetSpec(name='a', version='1.1', versioning='major_minor')
    spec.set_latest_version(['3', '2.1', '2.0', '1.3'])
    assert spec.version == '1.3'
