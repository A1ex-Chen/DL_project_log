def test_save_load_optional_components(self, expected_max_difference=0.0001):
    if not hasattr(self.pipeline_class, '_optional_components'):
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    for optional_component in pipe._optional_components:
        setattr(pipe, optional_component, None)
    generator_device = 'cpu'
    inputs = self.get_dummy_inputs(generator_device)
    output = pipe(**inputs).frames[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        pipe.save_pretrained(tmpdir, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
        for component in pipe_loaded.components.values():
            if hasattr(component, 'set_default_attn_processor'):
                component.set_default_attn_processor()
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)
    for optional_component in pipe._optional_components:
        self.assertTrue(getattr(pipe_loaded, optional_component) is None,
            f'`{optional_component}` did not stay set to None after loading.')
    inputs = self.get_dummy_inputs(generator_device)
    output_loaded = pipe_loaded(**inputs).frames[0]
    max_diff = np.abs(to_np(output) - to_np(output_loaded)).max()
    self.assertLess(max_diff, expected_max_difference)
