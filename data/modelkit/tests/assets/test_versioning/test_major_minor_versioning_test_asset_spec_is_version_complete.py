def test_asset_spec_is_version_complete():
    spec = AssetSpec(name='name', version='1.1', versioning='major_minor')
    assert spec.is_version_complete()
    spec = AssetSpec(name='name', version='1', versioning='major_minor')
    assert not spec.is_version_complete()
    spec = AssetSpec(name='name', versioning='major_minor')
    assert not spec.is_version_complete()
