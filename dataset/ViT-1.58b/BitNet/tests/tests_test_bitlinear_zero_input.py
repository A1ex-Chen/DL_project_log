def test_bitlinear_zero_input():
    layer = BitLinear(10, 20)
    input_tensor = torch.zeros(5, 10)
    output = layer(input_tensor)
    assert torch.allclose(output, torch.zeros(5, 20), atol=0.01)
