@pytest.mark.parametrize('s, spec', get_string_spec(['1', '1.2', '12']))
def test_string_asset_spec(s, spec):
    assert AssetSpec.from_string(s) == AssetSpec(**spec)
    assert AssetSpec.from_string(s, versioning='major_minor') == AssetSpec(
        versioning='major_minor', **spec)
