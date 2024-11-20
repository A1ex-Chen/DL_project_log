def test_freeu_disabled(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['return_dict'] = False
    inputs['output_type'] = 'np'
    output = pipe(**inputs)[0]
    pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    pipe.disable_freeu()
    freeu_keys = {'s1', 's2', 'b1', 'b2'}
    for upsample_block in pipe.unet.up_blocks:
        for key in freeu_keys:
            assert getattr(upsample_block, key
                ) is None, f'Disabling of FreeU should have set {key} to None.'
    inputs = self.get_dummy_inputs(torch_device)
    inputs['return_dict'] = False
    inputs['output_type'] = 'np'
    output_no_freeu = pipe(**inputs)[0]
    assert np.allclose(output, output_no_freeu, atol=0.01
        ), f'Disabling of FreeU should lead to results similar to the default pipeline results but Max Abs Error={np.abs(output_no_freeu - output).max()}.'
