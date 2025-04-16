def __init__(self, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel,
    unet: UNet2DConditionModel, scheduler: DDPMScheduler,
    image_noising_scheduler: DDPMScheduler, safety_checker: Optional[
    IFSafetyChecker], feature_extractor: Optional[CLIPImageProcessor],
    watermarker: Optional[IFWatermarker], requires_safety_checker: bool=True):
    super().__init__()
    if safety_checker is None and requires_safety_checker:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the IF license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    if safety_checker is not None and feature_extractor is None:
        raise ValueError(
            "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
    if unet.config.in_channels != 6:
        logger.warning(
            "It seems like you have loaded a checkpoint that shall not be used for super resolution from {unet.config._name_or_path} as it accepts {unet.config.in_channels} input channels instead of 6. Please make sure to pass a super resolution checkpoint as the `'unet'`: IFSuperResolutionPipeline.from_pretrained(unet=super_resolution_unet, ...)`."
            )
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        unet=unet, scheduler=scheduler, image_noising_scheduler=
        image_noising_scheduler, safety_checker=safety_checker,
        feature_extractor=feature_extractor, watermarker=watermarker)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
