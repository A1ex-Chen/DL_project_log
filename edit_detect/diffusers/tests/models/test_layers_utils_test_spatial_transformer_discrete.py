@require_torch_accelerator_with_fp64
def test_spatial_transformer_discrete(self):
    torch.manual_seed(0)
    backend_manual_seed(torch_device, 0)
    num_embed = 5
    sample = torch.randint(0, num_embed, (1, 32)).to(torch_device)
    spatial_transformer_block = Transformer2DModel(num_attention_heads=1,
        attention_head_dim=32, num_vector_embeds=num_embed, sample_size=16).to(
        torch_device).eval()
    with torch.no_grad():
        attention_scores = spatial_transformer_block(sample).sample
    assert attention_scores.shape == (1, num_embed - 1, 32)
    output_slice = attention_scores[0, -2:, -3:]
    expected_slice = torch.tensor([-1.7648, -1.0241, -2.0985, -1.8035, -
        1.6404, -1.2098], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
