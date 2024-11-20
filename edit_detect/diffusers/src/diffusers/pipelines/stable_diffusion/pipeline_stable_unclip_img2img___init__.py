def __init__(self, feature_extractor: CLIPImageProcessor, image_encoder:
    CLIPVisionModelWithProjection, image_normalizer:
    StableUnCLIPImageNormalizer, image_noising_scheduler:
    KarrasDiffusionSchedulers, tokenizer: CLIPTokenizer, text_encoder:
    CLIPTextModel, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, vae: AutoencoderKL):
    super().__init__()
    self.register_modules(feature_extractor=feature_extractor,
        image_encoder=image_encoder, image_normalizer=image_normalizer,
        image_noising_scheduler=image_noising_scheduler, tokenizer=
        tokenizer, text_encoder=text_encoder, unet=unet, scheduler=
        scheduler, vae=vae)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
