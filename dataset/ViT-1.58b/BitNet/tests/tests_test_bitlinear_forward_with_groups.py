@pytest.mark.parametrize('groups', [1, 2, 5])
def test_bitlinear_forward_with_groups(random_tensor, groups):
    layer = BitLinear(10, 20, groups=groups)
    output = layer(random_tensor)
    assert output.shape == (5, 20)
