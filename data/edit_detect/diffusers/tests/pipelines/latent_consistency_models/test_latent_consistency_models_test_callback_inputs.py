def test_callback_inputs(self):
    sig = inspect.signature(self.pipeline_class.__call__)
    if not ('callback_on_step_end_tensor_inputs' in sig.parameters and 
        'callback_on_step_end' in sig.parameters):
        return
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    self.assertTrue(hasattr(pipe, '_callback_tensor_inputs'),
        f' {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs'
        )

    def callback_inputs_test(pipe, i, t, callback_kwargs):
        missing_callback_inputs = set()
        for v in pipe._callback_tensor_inputs:
            if v not in callback_kwargs:
                missing_callback_inputs.add(v)
        self.assertTrue(len(missing_callback_inputs) == 0,
            f'Missing callback tensor inputs: {missing_callback_inputs}')
        last_i = pipe.num_timesteps - 1
        if i == last_i:
            callback_kwargs['denoised'] = torch.zeros_like(callback_kwargs[
                'denoised'])
        return callback_kwargs
    inputs = self.get_dummy_inputs(torch_device)
    inputs['callback_on_step_end'] = callback_inputs_test
    inputs['callback_on_step_end_tensor_inputs'] = pipe._callback_tensor_inputs
    inputs['output_type'] = 'latent'
    output = pipe(**inputs)[0]
    assert output.abs().sum() == 0
