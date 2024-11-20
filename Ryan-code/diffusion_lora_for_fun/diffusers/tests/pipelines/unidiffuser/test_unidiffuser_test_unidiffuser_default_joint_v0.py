def test_unidiffuser_default_joint_v0(self):
    device = 'cpu'
    components = self.get_dummy_components()
    unidiffuser_pipe = UniDiffuserPipeline(**components)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_joint_mode()
    assert unidiffuser_pipe.mode == 'joint'
    inputs = self.get_dummy_inputs_with_latents(device)
    del inputs['prompt']
    del inputs['image']
    sample = unidiffuser_pipe(**inputs)
    image = sample.images
    text = sample.text
    assert image.shape == (1, 32, 32, 3)
    image_slice = image[0, -3:, -3:, -1]
    expected_img_slice = np.array([0.576, 0.627, 0.6571, 0.4965, 0.4638, 
        0.5663, 0.5254, 0.5068, 0.5716])
    assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.001
    expected_text_prefix = ' no no no '
    assert text[0][:10] == expected_text_prefix
