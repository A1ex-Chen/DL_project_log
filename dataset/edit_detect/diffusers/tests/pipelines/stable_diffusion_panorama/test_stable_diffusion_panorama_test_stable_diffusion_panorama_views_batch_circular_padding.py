def test_stable_diffusion_panorama_views_batch_circular_padding(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = StableDiffusionPanoramaPipeline(**components)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    output = sd_pipe(**inputs, circular_padding=True, view_batch_size=2)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.6127, 0.6299, 0.4595, 0.4051, 0.4543, 
        0.3925, 0.551, 0.5693, 0.5031])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
