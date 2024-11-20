@pytest.mark.parametrize('version,valid', [('2021-11-15T17-30-56Z', True),
    ('0000-00-00T00-00-00Z', True), ('9999-99-99T99-99-99Z', True), (
    '2021-11-15T17-30-56', False), ('21-11-15T17-30-56Z', False), ('', False)])
def test_check_version_valid(version, valid):
    if valid:
        SimpleDateAssetsVersioningSystem.check_version_valid(version)
        assert SimpleDateAssetsVersioningSystem().is_version_valid(version)
    else:
        with pytest.raises(errors.InvalidVersionError):
            SimpleDateAssetsVersioningSystem.check_version_valid(version)
        assert not SimpleDateAssetsVersioningSystem().is_version_valid(version)
