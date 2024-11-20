def test_bitlinear_weight_group_scaling():
    layer = BitLinear(10, 20, groups=5)
    weight = layer.weight.view(layer.groups, -1)
    beta = torch.abs(weight).sum(dim=1, keepdim=True) / (weight.shape[0] *
        weight.shape[1])
    scaled_weight = weight * beta
    assert torch.allclose(scaled_weight, layer.weight.view(20, 10))
