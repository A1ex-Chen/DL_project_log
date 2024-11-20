def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, low_res_scheduler:
    DDPMScheduler, scheduler: KarrasDiffusionSchedulers, safety_checker:
    Optional[Any]=None, feature_extractor: Optional[CLIPImageProcessor]=
    None, watermarker: Optional[Any]=None, max_noise_level: int=350):
    super().__init__()
    if hasattr(vae, 'config'):
        is_vae_scaling_factor_set_to_0_08333 = hasattr(vae.config,
            'scaling_factor') and vae.config.scaling_factor == 0.08333
        if not is_vae_scaling_factor_set_to_0_08333:
            deprecation_message = (
                f"The configuration file of the vae does not contain `scaling_factor` or it is set to {vae.config.scaling_factor}, which seems highly unlikely. If your checkpoint is a fine-tuned version of `stabilityai/stable-diffusion-x4-upscaler` you should change 'scaling_factor' to 0.08333 Please make sure to update the config accordingly, as not doing so might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull Request for the `vae/config.json` file"
                )
            deprecate('wrong scaling_factor', '1.0.0', deprecation_message,
                standard_warn=False)
            vae.register_to_config(scaling_factor=0.08333)
    self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=
        tokenizer, unet=unet, low_res_scheduler=low_res_scheduler,
        scheduler=scheduler, safety_checker=safety_checker, watermarker=
        watermarker, feature_extractor=feature_extractor)
    self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
    self.image_processor = VaeImageProcessor(vae_scale_factor=self.
        vae_scale_factor, resample='bicubic')
    self.register_to_config(max_noise_level=max_noise_level)
