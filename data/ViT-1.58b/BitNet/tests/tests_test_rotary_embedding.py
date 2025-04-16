def test_rotary_embedding(bitnet_model, random_tensor):
    block = ParallelTransformerBlock(512, 64, 8, 4)
    rotary_emb1 = block.get_rotary_embedding(100, random_tensor.device)
    rotary_emb2 = block.get_rotary_embedding(200, random_tensor.device)
    assert rotary_emb1.shape == (100, 64)
    assert rotary_emb2.shape == (200, 64)
