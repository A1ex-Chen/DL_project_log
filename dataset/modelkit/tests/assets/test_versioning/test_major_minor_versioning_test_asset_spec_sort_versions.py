@pytest.mark.parametrize('version_list, result', TEST_CASES_SORT)
def test_asset_spec_sort_versions(version_list, result):
    spec = AssetSpec(name='name', versioning='major_minor')
    assert spec.sort_versions(version_list) == result
