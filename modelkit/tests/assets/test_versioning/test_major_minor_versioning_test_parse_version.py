@pytest.mark.parametrize('version, valid, values', TEST_CASES_PARSE)
def test_parse_version(version, valid, values):
    if version is None:
        return
    if valid and version != '':
        assert MajorMinorAssetsVersioningSystem._parse_version(version
            ) == values
    else:
        with pytest.raises(errors.InvalidVersionError):
            MajorMinorAssetsVersioningSystem._parse_version(version)
