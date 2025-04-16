@pytest.mark.parametrize('dim, dim_head, heads, ff_mult', [(512, 64, 8, 4),
    (256, 32, 4, 2), (128, 16, 2, 1)])
def test_parallel_transformer_block_raises_for_incorrect_input(dim,
    dim_head, heads, ff_mult):
    block = ParallelTransformerBlock(dim, dim_head, heads, ff_mult)
    with pytest.raises(Exception):
        block(torch.randn(32, 100))
