def test_get_initial_version():
    assert MajorMinorAssetsVersioningSystem.get_initial_version() == '0.0'
    MajorMinorAssetsVersioningSystem.check_version_valid(
        MajorMinorAssetsVersioningSystem.get_initial_version())
