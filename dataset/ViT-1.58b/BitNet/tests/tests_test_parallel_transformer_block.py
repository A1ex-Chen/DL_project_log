@pytest.mark.parametrize('dim, dim_head, heads, ff_mult', [(512, 64, 8, 4),
    (256, 32, 4, 2), (128, 16, 2, 1)])
def test_parallel_transformer_block(dim, dim_head, heads, ff_mult,
    random_tensor):
    block = ParallelTransformerBlock(dim, dim_head, heads, ff_mult)
    output = block(random_tensor)
    assert output.shape == random_tensor.shape
