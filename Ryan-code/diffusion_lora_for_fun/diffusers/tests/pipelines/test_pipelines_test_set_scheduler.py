def test_set_scheduler(self):
    unet = self.dummy_cond_unet()
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
    assert isinstance(sd.scheduler, DDIMScheduler)
    sd.scheduler = DDPMScheduler.from_config(sd.scheduler.config)
    assert isinstance(sd.scheduler, DDPMScheduler)
    sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
    assert isinstance(sd.scheduler, PNDMScheduler)
    sd.scheduler = LMSDiscreteScheduler.from_config(sd.scheduler.config)
    assert isinstance(sd.scheduler, LMSDiscreteScheduler)
    sd.scheduler = EulerDiscreteScheduler.from_config(sd.scheduler.config)
    assert isinstance(sd.scheduler, EulerDiscreteScheduler)
    sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler
        .config)
    assert isinstance(sd.scheduler, EulerAncestralDiscreteScheduler)
    sd.scheduler = DPMSolverMultistepScheduler.from_config(sd.scheduler.config)
    assert isinstance(sd.scheduler, DPMSolverMultistepScheduler)
