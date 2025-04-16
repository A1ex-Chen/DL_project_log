def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, image_encoder: CLIPVisionModelWithProjection
    =None, feature_extractor: CLIPImageProcessor=None,
    force_zeros_for_empty_prompt: bool=True, add_watermarker: Optional[bool
    ]=None):
    self.register_modules(vae=vae, text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=
        tokenizer_2, unet=unet, scheduler=scheduler, image_encoder=
        image_encoder, feature_extractor=feature_extractor)
    self.register_to_config(force_zeros_for_empty_prompt=
        force_zeros_for_empty_prompt)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.default_sample_size = self.unet.config.sample_size
    add_watermarker = (add_watermarker if add_watermarker is not None else
        is_invisible_watermark_available())
    if add_watermarker:
        self.watermark = StableDiffusionXLWatermarker()
    else:
        self.watermark = None
