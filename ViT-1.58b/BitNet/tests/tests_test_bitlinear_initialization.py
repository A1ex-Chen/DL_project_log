def test_bitlinear_initialization():
    layer = BitLinear(10, 20)
    assert layer.in_features == 10
    assert layer.out_features == 20
    assert layer.weight.shape == (20, 10)
