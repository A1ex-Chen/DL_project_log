@pytest.mark.parametrize('version_list, major, valid, result',
    TEST_CASES_FILTER)
def test_filter_versions(version_list, major, valid, result):
    if valid:
        assert list(MajorMinorAssetsVersioningSystem.filter_versions(
            version_list, major)) == result
    else:
        with pytest.raises(InvalidMajorVersionError):
            MajorMinorAssetsVersioningSystem.filter_versions(version_list,
                major)
