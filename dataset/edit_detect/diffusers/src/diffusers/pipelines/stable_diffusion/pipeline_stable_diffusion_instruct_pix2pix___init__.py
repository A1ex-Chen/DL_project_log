def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker,
    feature_extractor: CLIPImageProcessor, image_encoder: Optional[
    CLIPVisionModelWithProjection]=None, requires_safety_checker: bool=True):
    super().__init__()
    if safety_checker is None and requires_safety_checker:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    if safety_checker is not None and feature_extractor is None:
        raise ValueError(
            "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor, image_encoder=
        image_encoder)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
