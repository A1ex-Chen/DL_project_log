@parameterized.expand([[83, 4, [-0.2695, -0.1669, 0.0073, -0.3181, -0.1187,
    -0.1676, -0.1395, -0.5972]], [17, 0.55, [-0.129, -0.2588, 0.0551, -
    0.0916, 0.3286, 0.0238, -0.3669, 0.0322]], [8, 0.89, [-0.5283, 0.1198, 
    0.087, -0.1141, 0.9189, -0.015, 0.5474, 0.4319]], [3, 1000, [-0.5601, 
    0.2411, -0.5435, 0.1268, 1.1338, -0.2427, -0.028, -1.002]]])
@require_torch_accelerator_with_fp16
def test_compvis_sd_v1_5_fp16(self, seed, timestep, expected_slice):
    model = self.get_unet_model(model_id='runwayml/stable-diffusion-v1-5',
        fp16=True)
    latents = self.get_latents(seed, fp16=True)
    encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)
    timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)
    with torch.no_grad():
        sample = model(latents, timestep=timestep, encoder_hidden_states=
            encoder_hidden_states).sample
    assert sample.shape == latents.shape
    output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
    expected_output_slice = torch.tensor(expected_slice)
    assert torch_all_close(output_slice, expected_output_slice, atol=0.005)
