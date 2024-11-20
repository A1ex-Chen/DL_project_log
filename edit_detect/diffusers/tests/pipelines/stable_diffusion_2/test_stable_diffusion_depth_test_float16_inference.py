@unittest.skipIf(torch_device != 'cuda', reason='float16 requires CUDA')
def test_float16_inference(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    for name, module in components.items():
        if hasattr(module, 'half'):
            components[name] = module.half()
    pipe_fp16 = self.pipeline_class(**components)
    pipe_fp16.to(torch_device)
    pipe_fp16.set_progress_bar_config(disable=None)
    output = pipe(**self.get_dummy_inputs(torch_device))[0]
    output_fp16 = pipe_fp16(**self.get_dummy_inputs(torch_device))[0]
    max_diff = np.abs(output - output_fp16).max()
    self.assertLess(max_diff, 0.013,
        'The outputs of the fp16 and fp32 pipelines are too different.')
