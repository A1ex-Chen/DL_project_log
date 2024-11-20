@require_torch_gpu
def test_pipe_false_offload_warn(self):
    unet = self.dummy_cond_unet()
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
    sd.enable_model_cpu_offload()
    logger = logging.get_logger('diffusers.pipelines.pipeline_utils')
    with CaptureLogger(logger) as cap_logger:
        sd.to('cuda')
    assert 'It is strongly recommended against doing so' in str(cap_logger)
    sd = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=vae,
        text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=self.dummy_extractor)
