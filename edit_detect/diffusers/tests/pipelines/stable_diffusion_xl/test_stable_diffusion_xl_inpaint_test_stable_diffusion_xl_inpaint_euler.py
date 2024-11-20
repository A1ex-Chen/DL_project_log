def test_stable_diffusion_xl_inpaint_euler(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionXLInpaintPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.8029, 0.5523, 0.5825, 0.6003, 0.6702, 
        0.7018, 0.6369, 0.5955, 0.5123])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
