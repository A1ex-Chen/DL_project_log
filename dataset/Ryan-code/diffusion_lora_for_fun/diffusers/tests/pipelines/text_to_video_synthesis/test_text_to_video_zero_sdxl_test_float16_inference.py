@unittest.skipIf(torch_device != 'cuda', reason='float16 requires CUDA')
def test_float16_inference(self, expected_max_diff=0.05):
    components = self.get_dummy_components()
    for name, module in components.items():
        if hasattr(module, 'half'):
            components[name] = module.to(torch_device).half()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    components = self.get_dummy_components()
    pipe_fp16 = self.pipeline_class(**components)
    pipe_fp16.to(torch_device, torch.float16)
    pipe_fp16.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(self.generator_device)
    if 'generator' in inputs:
        inputs['generator'] = self.get_generator(self.generator_device)
    output = pipe(**inputs)[0]
    fp16_inputs = self.get_dummy_inputs(self.generator_device)
    if 'generator' in fp16_inputs:
        fp16_inputs['generator'] = self.get_generator(self.generator_device)
    output_fp16 = pipe_fp16(**fp16_inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_fp16)).max()
    self.assertLess(max_diff, expected_max_diff,
        'The outputs of the fp16 and fp32 pipelines are too different.')
