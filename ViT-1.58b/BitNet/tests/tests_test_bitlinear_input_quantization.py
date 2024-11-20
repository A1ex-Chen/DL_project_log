def test_bitlinear_input_quantization(random_tensor):
    layer = BitLinear(10, 20)
    quant_input, _ = absmax_quantize(random_tensor)
    output = layer(quant_input.float())
    assert output.shape == (5, 20)
