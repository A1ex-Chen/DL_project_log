def test_default_solver_type_after_switch(self):
    pipe = DiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-pipe', torch_dtype=torch
        .float16)
    assert pipe.scheduler.__class__ == DDIMScheduler
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    assert pipe.scheduler.config.solver_type == 'logrho'
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    assert pipe.scheduler.config.solver_type == 'bh2'
