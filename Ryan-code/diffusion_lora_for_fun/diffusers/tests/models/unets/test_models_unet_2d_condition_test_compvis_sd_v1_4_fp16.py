@parameterized.expand([[83, 4, [-0.2323, -0.1304, 0.0813, -0.3093, -0.0919,
    -0.1571, -0.1125, -0.5806]], [17, 0.55, [-0.0831, -0.2443, 0.0901, -
    0.0919, 0.3396, 0.0103, -0.3743, 0.0701]], [8, 0.89, [-0.4863, 0.0859, 
    0.0875, -0.1658, 0.9199, -0.0114, 0.4839, 0.4639]], [3, 1000, [-0.5649,
    0.2402, -0.5518, 0.1248, 1.1328, -0.2443, -0.0325, -1.0078]]])
@require_torch_accelerator_with_fp16
def test_compvis_sd_v1_4_fp16(self, seed, timestep, expected_slice):
    model = self.get_unet_model(model_id='CompVis/stable-diffusion-v1-4',
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
