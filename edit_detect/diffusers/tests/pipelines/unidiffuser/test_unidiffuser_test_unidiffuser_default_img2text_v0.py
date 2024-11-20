def test_unidiffuser_default_img2text_v0(self):
    device = 'cpu'
    components = self.get_dummy_components()
    unidiffuser_pipe = UniDiffuserPipeline(**components)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_image_to_text_mode()
    assert unidiffuser_pipe.mode == 'img2text'
    inputs = self.get_dummy_inputs_with_latents(device)
    del inputs['prompt']
    text = unidiffuser_pipe(**inputs).text
    expected_text_prefix = ' no no no '
    assert text[0][:10] == expected_text_prefix
