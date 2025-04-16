def test_bitlinear_no_bias():
    bitlinear = BitLinear(in_features=512, out_features=256, bias=False)
    assert bitlinear.bias is None
