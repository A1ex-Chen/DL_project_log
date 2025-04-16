def test_stable_diffusion_unflawed(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = DDIMScheduler.from_config(components[
        'scheduler'].config, timestep_spacing='trailing')
    sd_pipe = StableDiffusionPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['guidance_rescale'] = 0.7
    inputs['num_inference_steps'] = 10
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.4736, 0.5405, 0.4705, 0.4955, 0.5675, 
        0.4812, 0.531, 0.4967, 0.5064])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
