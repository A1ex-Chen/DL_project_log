def __init__(self, text_encoder_name, scheduler_name, unet_model_name=None,
    unet_model_config_path=None, snr_gamma=None, freeze_text_encoder=True,
    use_lora=False, ema_decay=0.999, teacher_guidance_scale=3, **kwargs):
    super().__init__()
    assert unet_model_name is not None or unet_model_config_path is not None, 'Either UNet pretrain model name or a config file path is required'
    self.text_encoder_name = text_encoder_name
    self.scheduler_name = scheduler_name
    self.unet_model_name = unet_model_name
    self.unet_model_config_path = unet_model_config_path
    self.freeze_text_encoder = freeze_text_encoder
    self.use_lora = use_lora
    self.ema_decay = ema_decay
    self.snr_gamma = snr_gamma
    logger.info(f'SNR gamma: {self.snr_gamma}')
    self.noise_scheduler = None
    self.teacher_guidance_scale = teacher_guidance_scale
    self.max_rand_guidance_scale = 6
    var_string = f'variable (max {self.max_rand_guidance_scale})'
    logger.info(
        f'Teacher guidance scale: {var_string if teacher_guidance_scale == -1 else teacher_guidance_scale}'
        )
    self.set_from = 'random'
    unet_config = UNet2DConditionModel.load_config(unet_model_config_path)
    self.teacher_unet = UNet2DConditionModel.from_config(unet_config,
        subfolder='unet')
    student_model_class = (UNet2DConditionGuidedModel if self.
        teacher_guidance_scale == -1 else UNet2DConditionModel)
    self.student_unet = student_model_class.from_config(unet_config,
        subfolder='unet')
    load_info = self.student_unet.load_state_dict(self.teacher_unet.
        state_dict(), strict=False)
    assert len(load_info.unexpected_keys
        ) == 0, f'Redundant keys in state_dict: {load_info.unexpected_keys}'
    self.student_ema_unet = deepcopy(self.student_unet)
    self.lightweight = 'light' in unet_model_config_path
    logger.info(f'Using the lightweight setting: {self.lightweight}')
    for model in [self.teacher_unet, self.student_ema_unet]:
        model.eval()
        model.requires_grad_(False)
    if 'stable-diffusion' in self.text_encoder_name:
        self.tokenizer = CLIPTokenizer.from_pretrained(self.
            text_encoder_name, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(self.
            text_encoder_name, subfolder='text_encoder')
    elif 't5' in self.text_encoder_name:
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = T5EncoderModel.from_pretrained(self.
            text_encoder_name)
    else:
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(self.text_encoder_name)
    if self.freeze_text_encoder:
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        logger.info('Text encoder is frozen.')
