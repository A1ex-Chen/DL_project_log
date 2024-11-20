@pytest.mark.parametrize('test, valid', [('_', False), ('a_', False), ('', 
    False), ('o', True), ('1', True), ('some_go0d_name', True), (
    'some_go/0d_name', True), ('SOME_GOOD_NAME_AS_WELL', True), (
    '50M3_G00D_N4ME_4S_W3LL', True), ('C:\\A\\L0cAL\\Windows\\file.ext', True)]
    )
def test_names(test, valid):
    if valid:
        AssetSpec.check_name_valid(test)
    else:
        with pytest.raises(errors.InvalidNameError):
            AssetSpec.check_name_valid(test)
