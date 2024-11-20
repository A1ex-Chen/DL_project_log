@parameterized.expand([[83, 4, [-0.1047, -1.7227, 0.1067, 0.0164, -0.5698, 
    -0.4172, -0.1388, 1.1387]], [17, 0.55, [0.0975, -0.2856, -0.3508, -0.46,
    0.3376, 0.293, -0.2747, -0.7026]], [8, 0.89, [-0.0952, 0.0183, -0.5825,
    -0.1981, 0.1131, 0.4668, -0.0395, -0.3486]], [3, 1000, [0.479, 0.4949, 
    -1.0732, -0.7158, 0.7959, -0.9478, 0.1105, -0.9741]]])
@require_torch_accelerator_with_fp16
def test_compvis_sd_inpaint_fp16(self, seed, timestep, expected_slice):
    model = self.get_unet_model(model_id=
        'runwayml/stable-diffusion-inpainting', fp16=True)
    latents = self.get_latents(seed, shape=(4, 9, 64, 64), fp16=True)
    encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)
    timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)
    with torch.no_grad():
        sample = model(latents, timestep=timestep, encoder_hidden_states=
            encoder_hidden_states).sample
    assert sample.shape == (4, 4, 64, 64)
    output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.005)
