def test_bitlinear_weight_sign():
    layer = BitLinear(10, 20)
    input_tensor = torch.randn(5, 10)
    output_before = layer(input_tensor)
    layer.weight.data = torch.abs(layer.weight.data)
    output_after = layer(input_tensor)
    assert not torch.allclose(output_before, output_after)
