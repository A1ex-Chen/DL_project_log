def test_pipe_same_device_id_offload(self):
    unet = self.dummy_cond_unet()
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    sd.enable_model_cpu_offload(gpu_id=5)
    assert sd._offload_gpu_id == 5
    sd.maybe_free_model_hooks()
    assert sd._offload_gpu_id == 5
