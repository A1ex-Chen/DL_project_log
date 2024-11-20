def __init__(self, tokenizer: CLIPTokenizer, image_feature_extractor:
    CLIPImageProcessor, text_encoder: CLIPTextModel, image_encoder:
    CLIPVisionModel, image_unet: UNet2DConditionModel, text_unet:
    UNet2DConditionModel, vae: AutoencoderKL, scheduler:
    KarrasDiffusionSchedulers):
    super().__init__()
    self.register_modules(tokenizer=tokenizer, image_feature_extractor=
        image_feature_extractor, text_encoder=text_encoder, image_encoder=
        image_encoder, image_unet=image_unet, text_unet=text_unet, vae=vae,
        scheduler=scheduler)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
