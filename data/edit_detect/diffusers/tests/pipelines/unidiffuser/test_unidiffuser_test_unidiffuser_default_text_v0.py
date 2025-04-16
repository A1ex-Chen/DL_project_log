def test_unidiffuser_default_text_v0(self):
    device = 'cpu'
    components = self.get_dummy_components()
    unidiffuser_pipe = UniDiffuserPipeline(**components)
    unidiffuser_pipe = unidiffuser_pipe.to(device)
    unidiffuser_pipe.set_progress_bar_config(disable=None)
    unidiffuser_pipe.set_text_mode()
    assert unidiffuser_pipe.mode == 'text'
    inputs = self.get_dummy_inputs(device)
    del inputs['prompt']
    del inputs['image']
    text = unidiffuser_pipe(**inputs).text
    expected_text_prefix = ' no no no '
    assert text[0][:10] == expected_text_prefix
