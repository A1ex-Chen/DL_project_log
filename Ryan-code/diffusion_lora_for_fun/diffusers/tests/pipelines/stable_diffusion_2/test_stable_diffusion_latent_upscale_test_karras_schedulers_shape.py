def test_karras_schedulers_shape(self):
    skip_schedulers = ['DDIMScheduler', 'DDPMScheduler', 'PNDMScheduler',
        'HeunDiscreteScheduler', 'EulerAncestralDiscreteScheduler',
        'KDPM2DiscreteScheduler', 'KDPM2AncestralDiscreteScheduler',
        'DPMSolverSDEScheduler', 'EDMEulerScheduler']
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.scheduler.register_to_config(skip_prk_steps=True)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(torch_device)
    inputs['num_inference_steps'] = 2
    outputs = []
    for scheduler_enum in KarrasDiffusionSchedulers:
        if scheduler_enum.name in skip_schedulers:
            continue
        scheduler_cls = getattr(diffusers, scheduler_enum.name)
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        output = pipe(**inputs)[0]
        outputs.append(output)
    assert check_same_shape(outputs)
