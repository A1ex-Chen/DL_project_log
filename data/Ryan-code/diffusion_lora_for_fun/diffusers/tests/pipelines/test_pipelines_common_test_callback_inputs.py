def test_callback_inputs(self):
    sig = inspect.signature(self.pipeline_class.__call__)
    has_callback_tensor_inputs = ('callback_on_step_end_tensor_inputs' in
        sig.parameters)
    has_callback_step_end = 'callback_on_step_end' in sig.parameters
    if not (has_callback_tensor_inputs and has_callback_step_end):
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    self.assertTrue(hasattr(pipe, '_callback_tensor_inputs'),
        f' {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs'
        )

    def callback_inputs_subset(pipe, i, t, callback_kwargs):
        for tensor_name, tensor_value in callback_kwargs.items():
            assert tensor_name in pipe._callback_tensor_inputs
        return callback_kwargs

    def callback_inputs_all(pipe, i, t, callback_kwargs):
        for tensor_name in pipe._callback_tensor_inputs:
            assert tensor_name in callback_kwargs
        for tensor_name, tensor_value in callback_kwargs.items():
            assert tensor_name in pipe._callback_tensor_inputs
        return callback_kwargs
    inputs = self.get_dummy_inputs(torch_device)
    inputs['callback_on_step_end'] = callback_inputs_subset
    inputs['callback_on_step_end_tensor_inputs'] = ['latents']
    inputs['output_type'] = 'latent'
    output = pipe(**inputs)[0]
    inputs['callback_on_step_end'] = callback_inputs_all
    inputs['callback_on_step_end_tensor_inputs'] = pipe._callback_tensor_inputs
    inputs['output_type'] = 'latent'
    output = pipe(**inputs)[0]

    def callback_inputs_change_tensor(pipe, i, t, callback_kwargs):
        is_last = i == pipe.num_timesteps - 1
        if is_last:
            callback_kwargs['latents'] = torch.zeros_like(callback_kwargs[
                'latents'])
        return callback_kwargs
    inputs['callback_on_step_end'] = callback_inputs_change_tensor
    inputs['callback_on_step_end_tensor_inputs'] = pipe._callback_tensor_inputs
    inputs['output_type'] = 'latent'
    output = pipe(**inputs)[0]
    assert output.abs().sum() == 0
