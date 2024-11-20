def test_stable_diffusion_xl_controlnet_img2img(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5557202, 0.46418434, 0.46983826, 0.623529,
        0.5557242, 0.49262643, 0.6070508, 0.5702978, 0.43777135])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
