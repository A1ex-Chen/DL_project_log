def test_bitlinear_forward():
    layer = BitLinear(10, 20)
    input_tensor = torch.randn(5, 10)
    output = layer(input_tensor)
    assert output.shape == (5, 20)
