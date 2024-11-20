@parameterized.expand([[33, 4, [-0.4424, 0.151, -0.1937, 0.2118, 0.3746, -
    0.3957, 0.016, -0.0435]], [47, 0.55, [-0.1508, 0.0379, -0.3075, 0.254, 
    0.3633, -0.0821, 0.1719, -0.0207]], [21, 0.89, [-0.6479, 0.6364, -
    0.3464, 0.8697, 0.4443, -0.6289, -0.0091, 0.1778]], [9, 1000, [0.8888, 
    -0.5659, 0.5834, -0.7469, 1.1912, -0.3923, 1.1241, -0.4424]]])
@require_torch_accelerator_with_fp16
def test_compvis_sd_v1_4(self, seed, timestep, expected_slice):
    model = self.get_unet_model(model_id='CompVis/stable-diffusion-v1-4')
    latents = self.get_latents(seed)
    encoder_hidden_states = self.get_encoder_hidden_states(seed)
    timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)
    with torch.no_grad():
        sample = model(latents, timestep=timestep, encoder_hidden_states=
            encoder_hidden_states).sample
    assert sample.shape == latents.shape
    output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.001)
