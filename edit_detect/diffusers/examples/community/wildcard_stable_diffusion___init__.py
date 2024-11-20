def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler: Union[
    DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler], safety_checker:
    StableDiffusionSafetyChecker, feature_extractor: CLIPImageProcessor):
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
    if safety_checker is None:
        logger.warning(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, safety_checker=
        safety_checker, feature_extractor=feature_extractor)
