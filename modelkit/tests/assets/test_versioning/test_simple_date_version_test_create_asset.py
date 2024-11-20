def test_create_asset():
    spec = AssetSpec(name='name', version='2020-11-15T17-30-56Z',
        versioning='simple_date')
    assert isinstance(spec.versioning, SimpleDateAssetsVersioningSystem)
    with pytest.raises(errors.InvalidVersionError):
        AssetSpec(name='name', version='2020-11-15T17-30-56', versioning=
            'simple_date')
