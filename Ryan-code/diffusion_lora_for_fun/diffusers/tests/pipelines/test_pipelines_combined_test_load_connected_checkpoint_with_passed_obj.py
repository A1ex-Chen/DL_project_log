def test_load_connected_checkpoint_with_passed_obj(self):
    pipeline = KandinskyV22CombinedPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-decoder')
    prior_scheduler = DDPMScheduler.from_config(pipeline.prior_scheduler.config
        )
    scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    assert pipeline.prior_scheduler.__class__ != prior_scheduler.__class__
    assert pipeline.scheduler.__class__ != scheduler.__class__
    pipeline_new = KandinskyV22CombinedPipeline.from_pretrained(
        'hf-internal-testing/tiny-random-kandinsky-v22-decoder',
        prior_scheduler=prior_scheduler, scheduler=scheduler)
    assert dict(pipeline_new.prior_scheduler.config) == dict(prior_scheduler
        .config)
    assert dict(pipeline_new.scheduler.config) == dict(scheduler.config)
