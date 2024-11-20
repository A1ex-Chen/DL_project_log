def test_control_guidance_switch(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.to(torch_device)
    scale = 10.0
    steps = 4
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = steps
    inputs['controlnet_conditioning_scale'] = scale
    output_1 = pipe(**inputs)[0]
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = steps
    inputs['controlnet_conditioning_scale'] = scale
    output_2 = pipe(**inputs, control_guidance_start=0.1,
        control_guidance_end=0.2)[0]
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = steps
    inputs['controlnet_conditioning_scale'] = scale
    output_3 = pipe(**inputs, control_guidance_start=[0.1],
        control_guidance_end=[0.2])[0]
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = steps
    inputs['controlnet_conditioning_scale'] = scale
    output_4 = pipe(**inputs, control_guidance_start=0.4,
        control_guidance_end=[0.5])[0]
    assert np.sum(np.abs(output_1 - output_2)) > 0.001
    assert np.sum(np.abs(output_1 - output_3)) > 0.001
    assert np.sum(np.abs(output_1 - output_4)) > 0.001
