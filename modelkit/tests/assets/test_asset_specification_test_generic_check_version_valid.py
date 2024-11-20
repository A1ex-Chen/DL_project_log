@pytest.mark.parametrize('test, valid', [('_', True), ('a_', True), ('a.a',
    True), ('1.a', True), ('a.1', True), ('.1', True), ('12.', True), ('', 
    True), ('1', True), ('12', True), ('12.1', True), ('12/1', False), (
    '12\x01', False)])
def test_generic_check_version_valid(test, valid):
    if valid:
        AssetSpec.check_version_valid(test)
    else:
        with pytest.raises(errors.InvalidVersionError):
            AssetSpec.check_version_valid(test)
