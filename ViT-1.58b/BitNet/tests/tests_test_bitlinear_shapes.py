@pytest.mark.parametrize('in_features,out_features', [(10, 20), (20, 40), (
    5, 10), (15, 10)])
def test_bitlinear_shapes(in_features, out_features):
    layer = BitLinear(in_features, out_features)
    assert layer.weight.shape == (out_features, in_features)
