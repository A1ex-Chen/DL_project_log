@pytest.mark.parametrize('groups', [1, 2, 5])
def test_bitlinear_groups(groups):
    layer = BitLinear(10, 20, groups=groups)
    assert layer.groups == groups
