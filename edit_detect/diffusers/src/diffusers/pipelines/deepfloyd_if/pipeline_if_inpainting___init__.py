def __init__(self, tokenizer: T5Tokenizer, text_encoder: T5EncoderModel,
    unet: UNet2DConditionModel, scheduler: DDPMScheduler, safety_checker:
    Optional[IFSafetyChecker], feature_extractor: Optional[
    CLIPImageProcessor], watermarker: Optional[IFWatermarker],
    requires_safety_checker: bool=True):
    super().__init__()
    if safety_checker is None and requires_safety_checker:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the IF license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    if safety_checker is not None and feature_extractor is None:
        raise ValueError(
            "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
    self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder,
        unet=unet, scheduler=scheduler, safety_checker=safety_checker,
        feature_extractor=feature_extractor, watermarker=watermarker)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
