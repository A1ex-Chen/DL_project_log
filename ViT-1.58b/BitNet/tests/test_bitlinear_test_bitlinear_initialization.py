def test_bitlinear_initialization():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=True)
    assert bitlinear.in_features == 512
    assert bitlinear.out_features == 256
    assert bitlinear.weight.shape == (256, 512)
    assert bitlinear.bias.shape == (256,)
    assert bitlinear.gamma.shape == (512,)
    assert bitlinear.beta.shape == (256,)
