def __init__(self, text_encoder_name, scheduler_name, unet_model_name=None,
    unet_model_config_path=None, snr_gamma=None, freeze_text_encoder=True,
    uncondition=False, use_edm=False, use_karras=False, use_lora=False,
    target_ema_decay=0.95, ema_decay=0.999, num_diffusion_steps=18,
    teacher_guidance_scale=1, vae=None, loss_type='clap'):
    assert loss_type == 'clap'
    super().__init__(text_encoder_name=text_encoder_name, scheduler_name=
        scheduler_name, unet_model_name=unet_model_name,
        unet_model_config_path=unet_model_config_path, snr_gamma=snr_gamma,
        freeze_text_encoder=freeze_text_encoder, uncondition=uncondition,
        use_edm=use_edm, use_karras=use_karras, use_lora=use_lora,
        target_ema_decay=target_ema_decay, ema_decay=ema_decay,
        num_diffusion_steps=num_diffusion_steps, teacher_guidance_scale=
        teacher_guidance_scale, vae=vae, loss_type=loss_type)
    self.ema_vae_decoder = deepcopy(self.vae.decoder)
    self.ema_vae_decoder.requires_grad_(False)
    self.ema_vae_decoder.eval()
    self.ema_vae_pqconv = deepcopy(self.vae.post_quant_conv)
    self.ema_vae_pqconv.requires_grad_(False)
    self.vae.decoder.train(self.training)
    self.vae.decoder.requires_grad_(True)
    self.vae.post_quant_conv.requires_grad_(True)
    self.vae.ema_decoder = self.ema_vae_decoder
    self.vae.ema_post_quant_conv = self.ema_vae_pqconv
    logger.info('Fine-tuning the VAE weights in addition to the U-Net.')
