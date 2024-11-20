def test_callback_cfg(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)

    def callback_no_cfg(pipe, i, t, callback_kwargs):
        if i == 1:
            for k, w in callback_kwargs.items():
                if k in self.callback_cfg_params:
                    callback_kwargs[k] = callback_kwargs[k].chunk(3)[0]
            pipe._guidance_scale = 1.0
        return callback_kwargs
    inputs = self.get_dummy_inputs(torch_device)
    inputs['guidance_scale'] = 1.0
    inputs['num_inference_steps'] = 2
    out_no_cfg = pipe(**inputs)[0]
    inputs['guidance_scale'] = 7.5
    inputs['callback_on_step_end'] = callback_no_cfg
    inputs['callback_on_step_end_tensor_inputs'] = pipe._callback_tensor_inputs
    out_callback_no_cfg = pipe(**inputs)[0]
    assert out_no_cfg.shape == out_callback_no_cfg.shape
