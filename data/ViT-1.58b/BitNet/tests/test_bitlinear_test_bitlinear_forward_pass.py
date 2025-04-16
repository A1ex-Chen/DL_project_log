def test_bitlinear_forward_pass():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=True)
    x = torch.randn(1, 512)
    out = bitlinear(x)
    assert out.shape == (1, 256)
