def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: Union[
    DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler], safety_checker:
    StableDiffusionSafetyChecker, feature_extractor: CLIPFeatureExtractor):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor)
