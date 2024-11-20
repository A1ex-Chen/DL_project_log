def test_callback_cfg(self):
    sig = inspect.signature(self.pipeline_class.__call__)
    has_callback_tensor_inputs = ('callback_on_step_end_tensor_inputs' in
        sig.parameters)
    has_callback_step_end = 'callback_on_step_end' in sig.parameters
    if not (has_callback_tensor_inputs and has_callback_step_end):
        return
    if 'guidance_scale' not in sig.parameters:
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    self.assertTrue(hasattr(pipe, '_callback_tensor_inputs'),
        f' {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs'
        )

    def callback_increase_guidance(pipe, i, t, callback_kwargs):
        pipe._guidance_scale += 1.0
        return callback_kwargs
    inputs = self.get_dummy_inputs(torch_device)
    inputs['guidance_scale'] = 2.0
    inputs['callback_on_step_end'] = callback_increase_guidance
    inputs['callback_on_step_end_tensor_inputs'] = pipe._callback_tensor_inputs
    _ = pipe(**inputs)[0]
    assert pipe.guidance_scale == inputs['guidance_scale'] + pipe.num_timesteps
