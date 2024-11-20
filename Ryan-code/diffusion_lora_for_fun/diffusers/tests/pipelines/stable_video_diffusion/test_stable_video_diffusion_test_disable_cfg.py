def test_disable_cfg(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator_device = 'cpu'
    inputs = self.get_dummy_inputs(generator_device)
    inputs['max_guidance_scale'] = 1.0
    output = pipe(**inputs).frames
    self.assertEqual(len(output.shape), 5)
