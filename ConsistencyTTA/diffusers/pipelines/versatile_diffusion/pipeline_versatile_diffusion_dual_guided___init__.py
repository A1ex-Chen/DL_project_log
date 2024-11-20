def __init__(self, tokenizer: CLIPTokenizer, image_feature_extractor:
    CLIPImageProcessor, text_encoder: CLIPTextModelWithProjection,
    image_encoder: CLIPVisionModelWithProjection, image_unet:
    UNet2DConditionModel, text_unet: UNetFlatConditionModel, vae:
    AutoencoderKL, scheduler: KarrasDiffusionSchedulers):
    super().__init__()
    self.register_modules(tokenizer=tokenizer, image_feature_extractor=
        image_feature_extractor, text_encoder=text_encoder, image_encoder=
        image_encoder, image_unet=image_unet, text_unet=text_unet, vae=vae,
        scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    if self.text_unet is not None and ('dual_cross_attention' not in self.
        image_unet.config or not self.image_unet.config.dual_cross_attention):
        self._convert_to_dual_attention()
