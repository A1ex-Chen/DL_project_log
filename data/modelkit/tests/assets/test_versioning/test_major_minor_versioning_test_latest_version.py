@pytest.mark.parametrize('version_list, major, valid, result',
    TEST_CASES_LATEST)
def test_latest_version(version_list, major, valid, result):
    if valid:
        assert MajorMinorAssetsVersioningSystem.latest_version(version_list,
            major) == result
    else:
        with pytest.raises(MajorVersionDoesNotExistError):
            MajorMinorAssetsVersioningSystem.latest_version(version_list, major
                )
