def test_bitnet_transformer_embed(bitnet_model):
    tokens = torch.randint(0, 20000, (1, 512))
    embedded = bitnet_model.emb(tokens)
    assert embedded.shape == (1, 512, 512)
