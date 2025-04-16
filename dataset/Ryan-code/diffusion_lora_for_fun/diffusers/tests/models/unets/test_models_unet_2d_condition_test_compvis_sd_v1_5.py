@parameterized.expand([[33, 4, [-0.443, 0.157, -0.1867, 0.2376, 0.3205, -
    0.3681, 0.0525, -0.0722]], [47, 0.55, [-0.1415, 0.0129, -0.3136, 0.2257,
    0.343, -0.0536, 0.2114, -0.0436]], [21, 0.89, [-0.7091, 0.6664, -0.3643,
    0.9032, 0.4499, -0.6541, 0.0139, 0.175]], [9, 1000, [0.8878, -0.5659, 
    0.5844, -0.7442, 1.1883, -0.3927, 1.1192, -0.4423]]])
@require_torch_accelerator
@skip_mps
def test_compvis_sd_v1_5(self, seed, timestep, expected_slice):
    model = self.get_unet_model(model_id='runwayml/stable-diffusion-v1-5')
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
