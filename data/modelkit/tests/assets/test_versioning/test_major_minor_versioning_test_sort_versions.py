@pytest.mark.parametrize('version_list, result', TEST_CASES_SORT)
def test_sort_versions(version_list, result):
    assert MajorMinorAssetsVersioningSystem.sort_versions(version_list
        ) == result
