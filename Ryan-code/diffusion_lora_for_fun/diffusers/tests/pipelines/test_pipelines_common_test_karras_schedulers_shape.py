def test_karras_schedulers_shape(self, num_inference_steps_for_strength=4,
    num_inference_steps_for_strength_for_iterations=5):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.scheduler.register_to_config(skip_prk_steps=True)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 2
    if 'strength' in inputs:
        inputs['num_inference_steps'] = num_inference_steps_for_strength
        inputs['strength'] = 0.5
    outputs = []
    for scheduler_enum in KarrasDiffusionSchedulers:
        if 'KDPM2' in scheduler_enum.name:
            inputs['num_inference_steps'
                ] = num_inference_steps_for_strength_for_iterations
        scheduler_cls = getattr(diffusers, scheduler_enum.name)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        output = pipe(**inputs)[0]
        outputs.append(output)
        if 'KDPM2' in scheduler_enum.name:
            inputs['num_inference_steps'] = 2
    assert check_same_shape(outputs)
