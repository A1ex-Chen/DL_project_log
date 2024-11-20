@pytest.mark.parametrize('spec_dict, valid', TEST_SPECS)
def test_create_asset(spec_dict, valid):
    if valid:
        AssetSpec(**spec_dict)
        AssetSpec(**spec_dict, versioning='major_minor')
    else:
        with pytest.raises(errors.InvalidAssetSpecError):
            AssetSpec(**spec_dict)
