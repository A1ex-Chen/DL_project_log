def test_ledits_pp_inversion(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = LEditsPPPipelineStableDiffusionXL(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inversion_inputs(device)
    inputs['image'] = inputs['image'][0]
    sd_pipe.invert(**inputs)
    assert sd_pipe.init_latents.shape == (1, 4, int(32 / sd_pipe.
        vae_scale_factor), int(32 / sd_pipe.vae_scale_factor))
    latent_slice = sd_pipe.init_latents[0, -1, -3:, -3:].to(device)
    expected_slice = np.array([-0.9084, -0.0367, 0.294, 0.0839, 0.689, 
        0.2651, -0.7103, 2.109, -0.7821])
    assert np.abs(latent_slice.flatten() - expected_slice).max() < 0.001
