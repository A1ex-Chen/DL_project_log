@parameterized.expand([[33, 4, [-0.7639, 0.0106, -0.1615, -0.3487, -0.0423,
    -0.7972, 0.0085, -0.4858]], [47, 0.55, [-0.6564, 0.0795, -1.9026, -
    0.6258, 1.8235, 1.2056, 1.2169, 0.9073]], [21, 0.89, [0.0327, 0.4399, -
    0.6358, 0.3417, 0.412, -0.5621, -0.0397, -1.043]], [9, 1000, [0.16, 
    0.7303, -1.0556, -0.3515, -0.744, -1.2037, -1.8149, -1.8931]]])
@require_torch_accelerator
@skip_mps
def test_compvis_sd_inpaint(self, seed, timestep, expected_slice):
    model = self.get_unet_model(model_id='runwayml/stable-diffusion-inpainting'
        )
    latents = self.get_latents(seed, shape=(4, 9, 64, 64))
    encoder_hidden_states = self.get_encoder_hidden_states(seed)
    timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)
    with torch.no_grad():
        sample = model(latents, timestep=timestep, encoder_hidden_states=
            encoder_hidden_states).sample
    assert sample.shape == (4, 4, 64, 64)
    output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.003)
