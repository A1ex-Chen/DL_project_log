def test_absmax_quantize():
    tensor = torch.tensor([1.5, -2.0, 3.0, -4.0])
    quant, dequant = absmax_quantize(tensor)
    assert quant.dtype == torch.int8
    assert torch.allclose(dequant, tensor, atol=0.01)
