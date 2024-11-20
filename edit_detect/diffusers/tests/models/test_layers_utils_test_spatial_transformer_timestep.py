def test_spatial_transformer_timestep(self):
    torch.manual_seed(0)
    backend_manual_seed(torch_device, 0)
    num_embeds_ada_norm = 5
    sample = torch.randn(1, 64, 64, 64).to(torch_device)
    spatial_transformer_block = Transformer2DModel(in_channels=64,
        num_attention_heads=2, attention_head_dim=32, dropout=0.0,
        cross_attention_dim=64, num_embeds_ada_norm=num_embeds_ada_norm).to(
        torch_device)
    with torch.no_grad():
        timestep_1 = torch.tensor(1, dtype=torch.long).to(torch_device)
        timestep_2 = torch.tensor(2, dtype=torch.long).to(torch_device)
        attention_scores_1 = spatial_transformer_block(sample, timestep=
            timestep_1).sample
        attention_scores_2 = spatial_transformer_block(sample, timestep=
            timestep_2).sample
    assert attention_scores_1.shape == (1, 64, 64, 64)
    assert attention_scores_2.shape == (1, 64, 64, 64)
    output_slice_1 = attention_scores_1[0, -1, -3:, -3:]
    output_slice_2 = attention_scores_2[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-0.3923, -1.0923, -1.7144, -1.557, 
        1.4154, 0.1738, -0.1157, -1.2998, -0.1703], device=torch_device)
    expected_slice_2 = torch.tensor([-0.4311, -1.1376, -1.7732, -1.5997, 
        1.345, 0.0964, -0.1569, -1.359, -0.2348], device=torch_device)
    assert torch.allclose(output_slice_1.flatten(), expected_slice, atol=0.001)
    assert torch.allclose(output_slice_2.flatten(), expected_slice_2, atol=
        0.001)
