def test_stable_diffusion_xl_euler(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5552, 0.5569, 0.4725, 0.4348, 0.4994, 
        0.4632, 0.5142, 0.5012, 0.47])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
