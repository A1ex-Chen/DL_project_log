def __init__(self, vae: AutoencoderKL, image_encoder:
    CLIPVisionModelWithProjection, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, safety_checker: StableDiffusionSafetyChecker,
    feature_extractor: CLIPImageProcessor, requires_safety_checker: bool=True):
    super().__init__()
    if safety_checker is None and requires_safety_checker:
        logger.warn(
            f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
    if safety_checker is not None and feature_extractor is None:
        raise ValueError(
            "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
    is_unet_version_less_0_9_0 = hasattr(unet.config, '_diffusers_version'
        ) and version.parse(version.parse(unet.config._diffusers_version).
        base_version) < version.parse('0.9.0.dev0')
    is_unet_sample_size_less_64 = hasattr(unet.config, 'sample_size'
        ) and unet.config.sample_size < 64
    if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
        deprecation_message = """The configuration file of the unet has set the default `sample_size` to smaller than 64 which seems highly unlikely .If you're checkpoint is a fine-tuned version of any of the following: 
- CompVis/stable-diffusion-v1-4 
- CompVis/stable-diffusion-v1-3 
- CompVis/stable-diffusion-v1-2 
- CompVis/stable-diffusion-v1-1 
- runwayml/stable-diffusion-v1-5 
- runwayml/stable-diffusion-inpainting 
 you should change 'sample_size' to 64 in the configuration file. Please make sure to update the config accordingly as leaving `sample_size=32` in the config might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `unet/config.json` file"""
        deprecate('sample_size<64', '1.0.0', deprecation_message,
            standard_warn=False)
        new_config = dict(unet.config)
        new_config['sample_size'] = 64
        unet._internal_dict = FrozenDict(new_config)
    self.register_modules(vae=vae, image_encoder=image_encoder, unet=unet,
        scheduler=scheduler, safety_checker=safety_checker,
        feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.register_to_config(requires_safety_checker=requires_safety_checker)
