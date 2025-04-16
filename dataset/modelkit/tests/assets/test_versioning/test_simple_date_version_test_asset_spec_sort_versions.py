def test_asset_spec_sort_versions():
    spec = AssetSpec(name='name', versioning='simple_date')
    version_list = ['2021-11-15T17-30-56Z', '2020-11-15T17-30-56Z',
        '2021-10-15T17-30-56Z']
    result = ['2021-11-15T17-30-56Z', '2021-10-15T17-30-56Z',
        '2020-11-15T17-30-56Z']
    assert spec.sort_versions(version_list) == result
