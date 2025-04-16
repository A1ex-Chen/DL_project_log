def test_unidiffuser_default_text2img_v0(self):
    device = 'cpu'
    components = self.get_dummy_components()
    unidiffuser_pipe = UniDiffuserPipeline(**components)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_text_to_image_mode()
    assert unidiffuser_pipe.mode == 'text2img'
    inputs = self.get_dummy_inputs_with_latents(device)
    del inputs['image']
    image = unidiffuser_pipe(**inputs).images
    assert image.shape == (1, 32, 32, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = np.array([0.5758, 0.6269, 0.657, 0.4967, 0.4639, 
        0.5664, 0.5257, 0.5067, 0.5715])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
