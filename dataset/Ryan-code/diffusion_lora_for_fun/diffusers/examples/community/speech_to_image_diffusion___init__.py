def __init__(self, speech_model: WhisperForConditionalGeneration,
    speech_processor: WhisperProcessor, vae: AutoencoderKL, text_encoder:
    CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet2DConditionModel,
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
    safety_checker: StableDiffusionSafetyChecker, feature_extractor:
    CLIPImageProcessor):
    super().__init__()
    if safety_checker is None:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    self.register_modules(speech_model=speech_model, speech_processor=
        speech_processor, vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, feature_extractor=
        feature_extractor)
