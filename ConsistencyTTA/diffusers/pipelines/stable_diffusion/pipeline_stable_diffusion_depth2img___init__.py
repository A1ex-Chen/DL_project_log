def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler:
    KarrasDiffusionSchedulers, depth_estimator: DPTForDepthEstimation,
    feature_extractor: DPTFeatureExtractor):
    super().__init__()
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
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, depth_estimator=
        depth_estimator, feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
