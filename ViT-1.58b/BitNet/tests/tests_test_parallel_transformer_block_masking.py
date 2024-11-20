def test_parallel_transformer_block_masking(random_tensor):
    block = ParallelTransformerBlock(512, 64, 8, 4)
    mask1 = block.get_mask(100, random_tensor.device)
    mask2 = block.get_mask(200, random_tensor.device)
    assert mask1.shape == (100, 100)
    assert mask2.shape == (200, 200)
