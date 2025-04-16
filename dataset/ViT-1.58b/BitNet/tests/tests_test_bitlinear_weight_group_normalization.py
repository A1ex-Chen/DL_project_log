@pytest.mark.parametrize('groups', [1, 2, 5])
def test_bitlinear_weight_group_normalization(groups):
    layer = BitLinear(10, 20, groups=groups)
    weight = layer.weight.view(groups, -1)
    mean = weight.mean(dim=1, keepdim=True)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=0.01)
