@pytest.mark.parametrize('test, valid', [('_', False), ('a_', False), ('', 
    True), (None, True), ('1', True), ('12', True)])
def test_check_version_number(test, valid):
    if valid:
        MajorMinorAssetsVersioningSystem._check_version_number(test)
    else:
        with pytest.raises(errors.InvalidVersionError):
            MajorMinorAssetsVersioningSystem._check_version_number(test)
