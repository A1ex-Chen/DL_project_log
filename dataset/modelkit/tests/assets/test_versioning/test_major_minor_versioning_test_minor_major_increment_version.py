@pytest.mark.parametrize('string, bump_major, major, valid, result',
    TEST_CASES_INCREMENT)
def test_minor_major_increment_version(string, bump_major, major, valid, result
    ):
    if valid:
        assert MajorMinorAssetsVersioningSystem.increment_version(string,
            params={'bump_major': bump_major, 'major': major}) == result
    else:
        with pytest.raises(MajorVersionDoesNotExistError):
            MajorMinorAssetsVersioningSystem.increment_version(string,
                params={'bump_major': bump_major, 'major': major})
