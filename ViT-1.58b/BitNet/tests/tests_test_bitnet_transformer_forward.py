def test_bitnet_transformer_forward(bitnet_model):
    tokens = torch.randint(0, 20000, (1, 512))
    logits = bitnet_model(tokens)
    assert logits.shape == (1, 20000)
