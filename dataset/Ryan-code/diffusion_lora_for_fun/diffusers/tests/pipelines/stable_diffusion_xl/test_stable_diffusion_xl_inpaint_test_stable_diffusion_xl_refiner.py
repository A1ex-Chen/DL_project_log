def test_stable_diffusion_xl_refiner(self):
    device = 'cpu'
    components = self.get_dummy_components(skip_first_text_encoder=True)
    sd_pipe = self.pipeline_class(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe(**inputs).images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.7045, 0.4838, 0.5454, 0.627, 0.6168, 
        0.6717, 0.6484, 0.5681, 0.4922])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
