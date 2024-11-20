@pytest.mark.parametrize('dim, depth, heads, dim_head, ff_mult', [(512, 6, 
    8, 64, 4), (256, 3, 4, 32, 2), (128, 2, 2, 16, 1)])
def test_transformer(dim, depth, heads, dim_head, ff_mult, random_tensor):
    transformer = Transformer(dim, depth, heads, dim_head, ff_mult)
    output = transformer(random_tensor)
    assert output.shape == random_tensor.shape
