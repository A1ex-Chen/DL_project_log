def test_unidiffuser_default_image_0(self):
    device = 'cpu'
    components = self.get_dummy_components()
    unidiffuser_pipe = UniDiffuserPipeline(**components)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_image_mode()
    assert unidiffuser_pipe.mode == 'img'
    inputs = self.get_dummy_inputs(device)
    del inputs['prompt']
    del inputs['image']
    image = unidiffuser_pipe(**inputs).images
    assert image.shape == (1, 32, 32, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.576, 0.627, 0.6571, 0.4966, 0.4638, 0.5663,
        0.5254, 0.5068, 0.5715])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
