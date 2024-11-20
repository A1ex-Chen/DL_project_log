def __init__(self, vae: AutoencoderKL, image_encoder:
    PaintByExampleImageEncoder, unet: UNet2DConditionModel, scheduler:
    Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    safety_checker: StableDiffusionSafetyChecker, feature_extractor:
    CLIPImageProcessor, requires_safety_checker: bool=False):
    super().__init__()
    self.register_modules(vae=vae, image_encoder=image_encoder, unet=unet,
        scheduler=scheduler, safety_checker=safety_checker,
        feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
