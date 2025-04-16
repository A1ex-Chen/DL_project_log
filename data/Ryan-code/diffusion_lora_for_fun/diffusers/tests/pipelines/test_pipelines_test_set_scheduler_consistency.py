def test_set_scheduler_consistency(self):
    unet = self.dummy_cond_unet()
    pndm = PNDMScheduler.from_config(
        'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
        'scheduler')
    ddim = DDIMScheduler.from_config(
        'hf-internal-testing/tiny-stable-diffusion-torch', subfolder=
        'scheduler')
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd = StableDiffusionPipeline(unet=unet, scheduler=pndm, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    pndm_config = sd.scheduler.config
    sd.scheduler = DDPMScheduler.from_config(pndm_config)
    sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
    pndm_config_2 = sd.scheduler.config
    pndm_config_2 = {k: v for k, v in pndm_config_2.items() if k in pndm_config
        }
    assert dict(pndm_config) == dict(pndm_config_2)
    sd = StableDiffusionPipeline(unet=unet, scheduler=ddim, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    ddim_config = sd.scheduler.config
    sd.scheduler = LMSDiscreteScheduler.from_config(ddim_config)
    sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
    ddim_config_2 = sd.scheduler.config
    ddim_config_2 = {k: v for k, v in ddim_config_2.items() if k in ddim_config
        }
    assert dict(ddim_config) == dict(ddim_config_2)
