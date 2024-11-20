@unittest.skip('Test is currently failing')
def test_float16_inference(self, expected_max_diff=0.05):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    components = self.get_dummy_components()
    pipe_fp16 = self.pipeline_class(**components)
    for component in pipe_fp16.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe_fp16.to(torch_device, torch.float16)
    pipe_fp16.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    output = pipe(**inputs).frames[0]
    fp16_inputs = self.get_dummy_inputs(torch_device)
    output_fp16 = pipe_fp16(**fp16_inputs).frames[0]
    max_diff = np.abs(to_np(output) - to_np(output_fp16)).max()
    self.assertLess(max_diff, expected_max_diff,
        'The outputs of the fp16 and fp32 pipelines are too different.')
