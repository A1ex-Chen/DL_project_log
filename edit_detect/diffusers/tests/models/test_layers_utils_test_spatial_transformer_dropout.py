def test_spatial_transformer_dropout(self):
    torch.manual_seed(0)
    backend_manual_seed(torch_device, 0)
    sample = torch.randn(1, 32, 64, 64).to(torch_device)
    spatial_transformer_block = Transformer2DModel(in_channels=32,
        num_attention_heads=2, attention_head_dim=16, dropout=0.3,
        cross_attention_dim=None).to(torch_device).eval()
    with torch.no_grad():
        attention_scores = spatial_transformer_block(sample).sample
    assert attention_scores.shape == (1, 32, 64, 64)
    output_slice = attention_scores[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-1.938, -0.0083, -1.3771, -1.5819, 
        0.5209, -0.6441, -1.8545, 0.7563, -0.9615], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
