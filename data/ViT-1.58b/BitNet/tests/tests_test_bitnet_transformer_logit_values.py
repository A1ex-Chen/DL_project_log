def test_bitnet_transformer_logit_values(bitnet_model):
    tokens = torch.randint(0, 20000, (1, 512))
    logits = bitnet_model(tokens)
    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0))
