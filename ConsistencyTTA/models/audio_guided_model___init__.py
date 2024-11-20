def __init__(self, text_encoder_name, scheduler_name, unet_model_name=None,
    unet_model_config_path=None, snr_gamma=None, freeze_text_encoder=True,
    use_lora=False, ema_decay=0.999, teacher_guidance_scale=3, **kwargs):
    super().__init__(text_encoder_name=text_encoder_name, scheduler_name=
        scheduler_name, unet_model_name=unet_model_name,
        unet_model_config_path=unet_model_config_path, snr_gamma=snr_gamma,
        freeze_text_encoder=freeze_text_encoder, use_lora=use_lora,
        ema_decay=ema_decay, teacher_guidance_scale=teacher_guidance_scale)
    self.noise_scheduler = DDPMScheduler.from_pretrained(self.
        scheduler_name, subfolder='scheduler')
    logger.info(
        f'Num noise schedule steps: {self.noise_scheduler.config.num_train_timesteps}'
        )
    logger.info(
        f'Noise scheduler prediction type: {self.noise_scheduler.config.prediction_type}'
        )
