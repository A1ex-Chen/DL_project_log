def test_bitlinear_quantization():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=True)
    x = torch.randn(1, 512)
    out = bitlinear(x)
    assert torch.all(out <= bitlinear.beta.unsqueeze(0).expand_as(out))
    assert torch.all(out >= -bitlinear.beta.unsqueeze(0).expand_as(out))
