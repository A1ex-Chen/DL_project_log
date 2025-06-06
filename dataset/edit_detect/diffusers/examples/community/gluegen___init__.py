def __init__(self, vae: AutoencoderKL, text_encoder: AutoModel, tokenizer:
    AutoTokenizer, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker,
    feature_extractor: CLIPImageProcessor, language_adapter: TranslatorNoLN
    =None, tensor_norm: torch.Tensor=None, requires_safety_checker: bool=True):
    super().__init__()
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor,
        language_adapter=language_adapter, tensor_norm=tensor_norm)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
