def test_spatial_transformer_cross_attention_dim(self):
    torch.manual_seed(0)
    backend_manual_seed(torch_device, 0)
    sample = torch.randn(1, 64, 64, 64).to(torch_device)
    spatial_transformer_block = Transformer2DModel(in_channels=64,
        num_attention_heads=2, attention_head_dim=32, dropout=0.0,
        cross_attention_dim=64).to(torch_device)
    with torch.no_grad():
        context = torch.randn(1, 4, 64).to(torch_device)
        attention_scores = spatial_transformer_block(sample, context).sample
    assert attention_scores.shape == (1, 64, 64, 64)
    output_slice = attention_scores[0, -1, -3:, -3:]
    expected_slice = torch.tensor([0.0143, -0.6909, -2.1547, -1.8893, 
        1.4097, 0.1359, -0.2521, -1.3359, 0.2598], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
