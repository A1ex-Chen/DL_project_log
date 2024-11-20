def test_asset_spec_set_latest_version():
    spec = AssetSpec(name='a', versioning='simple_date')
    spec.set_latest_version(['2021-11-15T17-31-06Z', '2021-11-14T18-00-00Z'])
    assert spec.version == '2021-11-15T17-31-06Z'
    spec = AssetSpec(name='a', version='2021-11-14T18-00-00Z', versioning=
        'simple_date')
    spec.set_latest_version(['2021-11-15T17-31-06Z', '2021-11-14T18-00-00Z'])
    assert spec.version == '2021-11-15T17-31-06Z'
