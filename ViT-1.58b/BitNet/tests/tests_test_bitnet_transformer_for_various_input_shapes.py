@pytest.mark.parametrize('batch_size, seq_len', [(1, 512), (32, 128), (64, 
    256)])
def test_bitnet_transformer_for_various_input_shapes(bitnet_model,
    batch_size, seq_len):
    tokens = torch.randint(0, 20000, (batch_size, seq_len))
    logits = bitnet_model(tokens)
    assert logits.shape == (batch_size, 20000)
