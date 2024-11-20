def __init__(self, vae, text_encoder, tokenizer, unet, scheduler,
    safety_checker, feature_extractor, requires_safety_checker: bool=True):
    super().__init__()
    logger.info(
        f'{self.__class__} is an experimntal pipeline and is likely to change in the future. We recommend to use this pipeline for fast experimentation / iteration if needed, but advice to rely on existing pipelines as defined in https://huggingface.co/docs/diffusers/api/schedulers#implemented-schedulers for production settings.'
        )
    scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor)
    model = ModelWrapper(unet, scheduler.alphas_cumprod)
    if scheduler.config.prediction_type == 'v_prediction':
        self.k_diffusion_model = CompVisVDenoiser(model)
    else:
        self.k_diffusion_model = CompVisDenoiser(model)
