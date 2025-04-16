def test_ledits_pp_inversion_batch(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = LEditsPPPipelineStableDiffusionXL(**components)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inversion_inputs(device)
    sd_pipe.invert(**inputs)
    assert sd_pipe.init_latents.shape == (2, 4, int(32 / sd_pipe.
        vae_scale_factor), int(32 / sd_pipe.vae_scale_factor))
    latent_slice = sd_pipe.init_latents[0, -1, -3:, -3:].to(device)
    print(latent_slice.flatten())
    expected_slice = np.array([0.2528, 0.1458, -0.2166, 0.4565, -0.5656, -
        1.0286, -0.9961, 0.5933, 1.1172])
    assert np.abs(latent_slice.flatten() - expected_slice).max() < 0.001
    latent_slice = sd_pipe.init_latents[1, -1, -3:, -3:].to(device)
    print(latent_slice.flatten())
    expected_slice = np.array([-0.0796, 2.0583, 0.55, 0.5358, 0.0282, -
        0.2803, -1.047, 0.7024, -0.0072])
    print(latent_slice.flatten())
    assert np.abs(latent_slice.flatten() - expected_slice).max() < 0.001
