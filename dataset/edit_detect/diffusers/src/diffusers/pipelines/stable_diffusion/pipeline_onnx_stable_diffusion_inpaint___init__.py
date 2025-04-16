def __init__(self, vae_encoder: OnnxRuntimeModel, vae_decoder:
    OnnxRuntimeModel, text_encoder: OnnxRuntimeModel, tokenizer:
    CLIPTokenizer, unet: OnnxRuntimeModel, scheduler: Union[DDIMScheduler,
    PNDMScheduler, LMSDiscreteScheduler], safety_checker: OnnxRuntimeModel,
    feature_extractor: CLIPImageProcessor, requires_safety_checker: bool=True):
    super().__init__()
    logger.info(
        '`OnnxStableDiffusionInpaintPipeline` is experimental and will very likely change in the future.'
        )
    if hasattr(scheduler.config, 'steps_offset'
        ) and scheduler.config.steps_offset != 1:
        deprecation_message = (
            f'The configuration file of this scheduler: {scheduler} is outdated. `steps_offset` should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure to update the config accordingly as leaving `steps_offset` might led to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            )
        deprecate('steps_offset!=1', '1.0.0', deprecation_message,
            standard_warn=False)
        new_config = dict(scheduler.config)
        new_config['steps_offset'] = 1
        scheduler._internal_dict = FrozenDict(new_config)
    if hasattr(scheduler.config, 'clip_sample'
        ) and scheduler.config.clip_sample is True:
        deprecation_message = (
            f'The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`. `clip_sample` should be set to False in the configuration file. Please make sure to update the config accordingly as not setting `clip_sample` in the config might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json` file'
            )
        deprecate('clip_sample not set', '1.0.0', deprecation_message,
            standard_warn=False)
        new_config = dict(scheduler.config)
        new_config['clip_sample'] = False
        scheduler._internal_dict = FrozenDict(new_config)
    if safety_checker is None and requires_safety_checker:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    if safety_checker is not None and feature_extractor is None:
        raise ValueError(
            "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
    self.register_modules(vae_encoder=vae_encoder, vae_decoder=vae_decoder,
        text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=scheduler, safety_checker=safety_checker,
        feature_extractor=feature_extractor)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
