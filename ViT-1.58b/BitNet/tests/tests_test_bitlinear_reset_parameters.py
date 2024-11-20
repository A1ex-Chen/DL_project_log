def test_bitlinear_reset_parameters():
    layer = BitLinear(10, 20)
    original_weights = layer.weight.clone()
    layer.reset_parameters()
    assert not torch.equal(original_weights, layer.weight)
