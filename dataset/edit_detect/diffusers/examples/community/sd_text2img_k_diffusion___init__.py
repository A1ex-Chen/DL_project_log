def __init__(self, vae, text_encoder, tokenizer, unet, scheduler,
    safety_checker, feature_extractor):
    super().__init__()
    if safety_checker is None:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor)
    model = ModelWrapper(unet, scheduler.alphas_cumprod)
    if scheduler.config.prediction_type == 'v_prediction':
        self.k_diffusion_model = CompVisVDenoiser(model)
    else:
        self.k_diffusion_model = CompVisDenoiser(model)
