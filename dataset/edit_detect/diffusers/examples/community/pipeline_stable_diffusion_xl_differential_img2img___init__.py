def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection, tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, image_encoder: CLIPVisionModelWithProjection
    =None, feature_extractor: CLIPImageProcessor=None,
    requires_aesthetics_score: bool=False, force_zeros_for_empty_prompt:
    bool=True, add_watermarker: Optional[bool]=None):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder,
        text_encoder_2=text_encoder_2, tokenizer=tokenizer, tokenizer_2=
        tokenizer_2, unet=unet, image_encoder=image_encoder,
        feature_extractor=feature_extractor, scheduler=scheduler)
    self.register_to_config(force_zeros_for_empty_prompt=
        force_zeros_for_empty_prompt)
    self.register_to_config(requires_aesthetics_score=requires_aesthetics_score
        )
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    add_watermarker = (add_watermarker if add_watermarker is not None else
        is_invisible_watermark_available())
    if add_watermarker:
        self.watermark = StableDiffusionXLWatermarker()
    else:
        self.watermark = None
