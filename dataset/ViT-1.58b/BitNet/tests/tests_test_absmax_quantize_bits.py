@pytest.mark.parametrize('bits', [4, 8, 12, 16])
def test_absmax_quantize_bits(random_tensor, bits):
    quant, dequant = absmax_quantize(random_tensor, bits=bits)
    assert quant.dtype == torch.int8
    assert torch.allclose(dequant, random_tensor, atol=0.01)
