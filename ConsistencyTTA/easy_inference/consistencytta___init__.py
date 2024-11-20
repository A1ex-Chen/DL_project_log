def __init__(self):
    super().__init__()
    unet_model_config_path = 'tango_diffusion_light.json'
    unet_config = UNet2DConditionGuidedModel.load_config(unet_model_config_path
        )
    self.unet = UNet2DConditionGuidedModel.from_config(unet_config,
        subfolder='unet')
    unet_weight_path = 'consistencytta_clapft_ckpt/unet_state_dict.pt'
    unet_weight_sd = torch.load(unet_weight_path, map_location='cpu')
    self.unet.load_state_dict(unet_weight_sd)
    text_encoder_name = 'google/flan-t5-large'
    self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
    self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name)
    self.text_encoder.eval()
    self.text_encoder.requires_grad_(False)
    raw_vae_path = 'consistencytta_clapft_ckpt/vae_state_dict.pt'
    raw_vae_sd = torch.load(raw_vae_path, map_location='cpu')
    vae_state_dict, scale_factor = raw_vae_sd['state_dict'], raw_vae_sd[
        'scale_factor']
    config = default_audioldm_config('audioldm-s-full')
    vae_config = config['model']['params']['first_stage_config']['params']
    vae_config['scale_factor'] = scale_factor
    self.vae = AutoencoderKL(**vae_config)
    self.vae.load_state_dict(vae_state_dict)
    self.vae.eval()
    self.vae.requires_grad_(False)
    self.fn_STFT = TacotronSTFT(config['preprocessing']['stft'][
        'filter_length'], config['preprocessing']['stft']['hop_length'],
        config['preprocessing']['stft']['win_length'], config[
        'preprocessing']['mel']['n_mel_channels'], config['preprocessing'][
        'audio']['sampling_rate'], config['preprocessing']['mel'][
        'mel_fmin'], config['preprocessing']['mel']['mel_fmax'])
    self.fn_STFT.eval()
    self.fn_STFT.requires_grad_(False)
    self.scheduler = HeunDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path='stabilityai/stable-diffusion-2-1',
        subfolder='scheduler')
