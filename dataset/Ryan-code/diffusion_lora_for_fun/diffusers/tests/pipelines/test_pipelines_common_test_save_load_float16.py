@unittest.skipIf(torch_device != 'cuda', reason='float16 requires CUDA')
def test_save_load_float16(self, expected_max_diff=0.01):
    components = self.get_dummy_components()
    for name, module in components.items():
        if hasattr(module, 'half'):
            components[name] = module.to(torch_device).half()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    output = pipe(**inputs)[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir)
        pipe_loaded = self.pipeline_class.from_pretrained(tmpdir,
            torch_dtype=torch.float16)
        for component in pipe_loaded.components.values():
            if hasattr(component, 'set_default_attn_processor'):
                component.set_default_attn_processor()
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)
    for name, component in pipe_loaded.components.items():
        if hasattr(component, 'dtype'):
            self.assertTrue(component.dtype == torch.float16,
                f'`{name}.dtype` switched from `float16` to {component.dtype} after loading.'
                )
    inputs = self.get_dummy_inputs(torch_device)
    output_loaded = pipe_loaded(**inputs)[0]
    max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
    self.assertLess(max_diff, expected_max_diff,
        'The output of the fp16 pipeline changed after saving and loading.')
