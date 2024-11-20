def __init__(self, vae: FlaxAutoencoderKL, text_encoder: FlaxCLIPTextModel,
    tokenizer: CLIPTokenizer, unet: FlaxUNet2DConditionModel, scheduler:
    Union[FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler,
    FlaxDPMSolverMultistepScheduler], safety_checker:
    FlaxStableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor,
    dtype: jnp.dtype=jnp.float32):
    super().__init__()
    self.dtype = dtype
    if safety_checker is None:
        logger.warn(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
