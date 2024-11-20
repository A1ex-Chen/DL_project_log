@pytest.mark.parametrize('s, spec', get_string_spec(['2021-11-14T18-00-00Z']))
def test_string_asset_spec(s, spec):
    assert AssetSpec.from_string(s, versioning='simple_date') == AssetSpec(
        versioning='simple_date', **spec)
