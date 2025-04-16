def __init__(self, text_encoder_name, scheduler_name, unet_model_name=None,
    unet_model_config_path=None, snr_gamma=None, freeze_text_encoder=True,
    uncondition=False, use_edm=False, use_karras=False, use_lora=False,
    target_ema_decay=0.95, ema_decay=0.999, num_diffusion_steps=18,
    teacher_guidance_scale=1, vae=None, loss_type='mse'):
    super().__init__(text_encoder_name=text_encoder_name, scheduler_name=
        scheduler_name, unet_model_name=unet_model_name,
        unet_model_config_path=unet_model_config_path, snr_gamma=snr_gamma,
        freeze_text_encoder=freeze_text_encoder, use_lora=use_lora,
        ema_decay=ema_decay, teacher_guidance_scale=teacher_guidance_scale)
    assert unet_model_name is not None or unet_model_config_path is not None, 'Either UNet pretrain model name or a config file path is required'
    self.uncondition = uncondition
    self.use_edm = use_edm
    self.use_karras = use_karras
    self.target_ema_decay = target_ema_decay
    self.num_diffusion_steps = num_diffusion_steps
    self.lightweight = 'light' in unet_model_config_path
    logger.info(f'Use the lightweight model setting: {self.lightweight}')
    self.student_target_unet = deepcopy(self.student_unet)
    self.student_target_unet.eval()
    self.student_target_unet.requires_grad_(False)
    sched_class = HeunDiscreteScheduler if self.use_edm else DDIMScheduler
    self.noise_scheduler = sched_class.from_pretrained(self.scheduler_name,
        subfolder='scheduler')
    if self.use_karras:
        if self.use_edm:
            logger.info('Using Karras noise schedule.')
            self.noise_scheduler.use_karras_sigmas = True
        else:
            ValueError(
                'Karras noise schedule can only be used with Heun scheduler.')
    self.noise_scheduler.set_timesteps(self.num_diffusion_steps, device=
        self.device)
    self.vae = vae
    self.loss_type = loss_type
    self.losses = {'mse': MSELoss(reduction='instance'), 'mel': MelLoss(vae
        =self.vae, reduction='instance'), 'stft': MultiResolutionSTFTLoss(
        vae=self.vae, reduction='instance', fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window=
        'hann_window', factor_sc=0.1, factor_mag=0.1, factor_mse=0.8),
        'clap': CLAPLoss(vae=self.vae, reduction='instance', mse_weight=1.0,
        clap_weight=0.1)}
    logger.info(f'Using the {self.loss_type} loss.')
    self.loss = self.losses[self.loss_type]
