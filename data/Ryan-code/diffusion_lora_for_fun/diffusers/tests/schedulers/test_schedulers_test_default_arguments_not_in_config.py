def test_default_arguments_not_in_config(self):
    pipe = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-pipe', torch_dtype=torch
        .float16)
    assert pipe.scheduler.__class__ == DDIMScheduler
    assert pipe.scheduler.config.timestep_spacing == 'leading'
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    assert pipe.scheduler.config.timestep_spacing == 'linspace'
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.
        config, timestep_spacing='trailing')
    assert pipe.scheduler.config.timestep_spacing == 'trailing'
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    assert pipe.scheduler.config.timestep_spacing == 'trailing'
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    assert pipe.scheduler.config.timestep_spacing == 'trailing'
