def test_cfg(self):
    sig = inspect.signature(self.pipeline_class.__call__)
    if 'guidance_scale' not in sig.parameters:
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['guidance_scale'] = 1.0
    out_no_cfg = pipe(**inputs)[0]
    inputs['guidance_scale'] = 7.5
    out_cfg = pipe(**inputs)[0]
    assert out_cfg.shape == out_no_cfg.shape
