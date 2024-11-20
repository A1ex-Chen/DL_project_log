@pytest.mark.parametrize('version, valid, values', TEST_CASES_PARSE)
def test_check_version_valid(version, valid, values):
    if valid:
        MajorMinorAssetsVersioningSystem.check_version_valid(version)
        assert MajorMinorAssetsVersioningSystem().is_version_valid(version)
    else:
        with pytest.raises(errors.InvalidVersionError):
            MajorMinorAssetsVersioningSystem.check_version_valid(version)
        assert not MajorMinorAssetsVersioningSystem().is_version_valid(version)
