def __init__(self, image_feature_extractor: CLIPImageProcessor,
    image_encoder: CLIPVisionModelWithProjection, image_unet:
    UNet2DConditionModel, vae: AutoencoderKL, scheduler:
    KarrasDiffusionSchedulers):
    super().__init__()
    self.register_modules(image_feature_extractor=image_feature_extractor,
        image_encoder=image_encoder, image_unet=image_unet, vae=vae,
        scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
