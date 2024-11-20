@skip_mps
def test_freeu_enabled(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['return_dict'] = False
    inputs['output_type'] = 'np'
    output = pipe(**inputs)[0]
    pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['return_dict'] = False
    inputs['output_type'] = 'np'
    output_freeu = pipe(**inputs)[0]
    assert not np.allclose(output[0, -3:, -3:, -1], output_freeu[0, -3:, -3
        :, -1]), 'Enabling of FreeU should lead to different results.'
