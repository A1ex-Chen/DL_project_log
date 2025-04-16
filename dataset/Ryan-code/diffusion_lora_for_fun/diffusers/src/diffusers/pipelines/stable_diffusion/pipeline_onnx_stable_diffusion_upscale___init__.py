def __init__(self, vae: OnnxRuntimeModel, text_encoder: OnnxRuntimeModel,
    tokenizer: Any, unet: OnnxRuntimeModel, low_res_scheduler:
    DDPMScheduler, scheduler: KarrasDiffusionSchedulers, safety_checker:
    Optional[OnnxRuntimeModel]=None, feature_extractor: Optional[
    CLIPImageProcessor]=None, max_noise_level: int=350, num_latent_channels
    =4, num_unet_input_channels=7, requires_safety_checker: bool=True):
    super().__init__()
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
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, low_res_scheduler=
        low_res_scheduler, safety_checker=safety_checker, feature_extractor
        =feature_extractor)
    self.register_to_config(max_noise_level=max_noise_level,
        num_latent_channels=num_latent_channels, num_unet_input_channels=
        num_unet_input_channels)
