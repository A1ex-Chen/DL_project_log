def test_versioning_values():
    AssetSpec(name='a')
    AssetSpec(name='a', versioning='major_minor')
    AssetSpec(name='a', versioning='simple_date')
    with pytest.raises(errors.UnknownAssetsVersioningSystemError):
        AssetSpec(name='a', versioning='unk_versioning')
