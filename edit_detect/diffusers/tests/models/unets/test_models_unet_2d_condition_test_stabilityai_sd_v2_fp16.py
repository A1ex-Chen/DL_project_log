@parameterized.expand([[83, 4, [0.1514, 0.0807, 0.1624, 0.1016, -0.1896, 
    0.0263, 0.0677, 0.231]], [17, 0.55, [0.1164, -0.0216, 0.017, 0.1589, -
    0.312, 0.1005, -0.0581, -0.1458]], [8, 0.89, [-0.1758, -0.0169, 0.1004,
    -0.1411, 0.1312, 0.1103, -0.1996, 0.2139]], [3, 1000, [0.1214, 0.0352, 
    -0.0731, -0.1562, -0.0994, -0.0906, -0.234, -0.0539]]])
@require_torch_accelerator_with_fp16
def test_stabilityai_sd_v2_fp16(self, seed, timestep, expected_slice):
    model = self.get_unet_model(model_id='stabilityai/stable-diffusion-2',
        fp16=True)
    latents = self.get_latents(seed, shape=(4, 4, 96, 96), fp16=True)
    encoder_hidden_states = self.get_encoder_hidden_states(seed, shape=(4, 
        77, 1024), fp16=True)
    timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)
    with torch.no_grad():
        sample = model(latents, timestep=timestep, encoder_hidden_states=
            encoder_hidden_states).sample
    assert sample.shape == latents.shape
    output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.005)
