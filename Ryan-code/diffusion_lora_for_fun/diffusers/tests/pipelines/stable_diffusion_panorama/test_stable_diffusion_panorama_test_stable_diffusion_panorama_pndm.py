def test_stable_diffusion_panorama_pndm(self):
    device = 'cpu'
    components = self.get_dummy_components()
    components['scheduler'] = PNDMScheduler(beta_start=0.00085, beta_end=
        0.012, beta_schedule='scaled_linear', skip_prk_steps=True)
    sd_pipe = StableDiffusionPanoramaPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.6391, 0.6291, 0.4861, 0.5134, 0.5552, 
        0.4578, 0.5032, 0.5023, 0.4539])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
