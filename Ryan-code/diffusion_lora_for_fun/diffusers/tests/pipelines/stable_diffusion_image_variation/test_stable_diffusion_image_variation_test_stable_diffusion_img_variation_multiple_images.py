def test_stable_diffusion_img_variation_multiple_images(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionImageVariationPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['image'] = 2 * [inputs['image']]
    output = sd_pipe(**inputs)
    image = output.images
    image_slice = image[-1, -3:, -3:, -1]
    assert image.shape == (2, 64, 64, 3)
    expected_slice = np.array([0.6892, 0.5637, 0.5836, 0.5771, 0.6254, 
        0.6409, 0.558, 0.5569, 0.5289])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
