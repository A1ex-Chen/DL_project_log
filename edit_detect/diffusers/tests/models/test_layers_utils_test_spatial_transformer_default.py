def test_spatial_transformer_default(self):
    torch.manual_seed(0)
    backend_manual_seed(torch_device, 0)
    sample = torch.randn(1, 32, 64, 64).to(torch_device)
    spatial_transformer_block = Transformer2DModel(in_channels=32,
        num_attention_heads=1, attention_head_dim=32, dropout=0.0,
        cross_attention_dim=None).to(torch_device)
    with torch.no_grad():
        attention_scores = spatial_transformer_block(sample).sample
    assert attention_scores.shape == (1, 32, 64, 64)
    output_slice = attention_scores[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-1.9455, -0.0066, -1.3933, -1.5878, 
        0.5325, -0.6486, -1.8648, 0.7515, -0.9689], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
