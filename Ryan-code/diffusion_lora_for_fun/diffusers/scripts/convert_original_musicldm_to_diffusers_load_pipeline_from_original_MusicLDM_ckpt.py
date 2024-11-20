def load_pipeline_from_original_MusicLDM_ckpt(checkpoint_path: str,
    original_config_file: str=None, image_size: int=1024, prediction_type:
    str=None, extract_ema: bool=False, scheduler_type: str='ddim',
    num_in_channels: int=None, model_channels: int=None, num_head_channels:
    int=None, device: str=None, from_safetensors: bool=False
    ) ->MusicLDMPipeline:
    """
    Load an MusicLDM pipeline object from a `.ckpt`/`.safetensors` file and (ideally) a `.yaml` config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path (`str`): Path to `.ckpt` file.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            set to the MusicLDM-s-full-v2 config.
        image_size (`int`, *optional*, defaults to 1024):
            The image size that the model was trained on.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. If `None`, will be automatically
            inferred by looking for a key in the config. For the default config, the prediction type is `'epsilon'`.
        num_in_channels (`int`, *optional*, defaults to None):
            The number of UNet input channels. If `None`, it will be automatically inferred from the config.
        model_channels (`int`, *optional*, defaults to None):
            The number of UNet model channels. If `None`, it will be automatically inferred from the config. Override
            to 128 for the small checkpoints, 192 for the medium checkpoints and 256 for the large.
        num_head_channels (`int`, *optional*, defaults to None):
            The number of UNet head channels. If `None`, it will be automatically inferred from the config. Override
            to 32 for the small and medium checkpoints, and 64 for the large.
        scheduler_type (`str`, *optional*, defaults to 'pndm'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.
        return: An MusicLDMPipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """
    if from_safetensors:
        from safetensors import safe_open
        checkpoint = {}
        with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    elif device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if original_config_file is None:
        original_config = DEFAULT_CONFIG
    else:
        original_config = yaml.safe_load(original_config_file)
    if num_in_channels is not None:
        original_config['model']['params']['unet_config']['params'][
            'in_channels'] = num_in_channels
    if model_channels is not None:
        original_config['model']['params']['unet_config']['params'][
            'model_channels'] = model_channels
    if num_head_channels is not None:
        original_config['model']['params']['unet_config']['params'][
            'num_head_channels'] = num_head_channels
    if 'parameterization' in original_config['model']['params'
        ] and original_config['model']['params']['parameterization'] == 'v':
        if prediction_type is None:
            prediction_type = 'v_prediction'
    elif prediction_type is None:
        prediction_type = 'epsilon'
    if image_size is None:
        image_size = 512
    num_train_timesteps = original_config['model']['params']['timesteps']
    beta_start = original_config['model']['params']['linear_start']
    beta_end = original_config['model']['params']['linear_end']
    scheduler = DDIMScheduler(beta_end=beta_end, beta_schedule=
        'scaled_linear', beta_start=beta_start, num_train_timesteps=
        num_train_timesteps, steps_offset=1, clip_sample=False,
        set_alpha_to_one=False, prediction_type=prediction_type)
    scheduler.register_to_config(clip_sample=False)
    if scheduler_type == 'pndm':
        config = dict(scheduler.config)
        config['skip_prk_steps'] = True
        scheduler = PNDMScheduler.from_config(config)
    elif scheduler_type == 'lms':
        scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == 'heun':
        scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == 'euler':
        scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
    elif scheduler_type == 'euler-ancestral':
        scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.
            config)
    elif scheduler_type == 'dpm':
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
    elif scheduler_type == 'ddim':
        scheduler = scheduler
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    unet_config = create_unet_diffusers_config(original_config, image_size=
        image_size)
    unet = UNet2DConditionModel(**unet_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint,
        unet_config, path=checkpoint_path, extract_ema=extract_ema)
    unet.load_state_dict(converted_unet_checkpoint)
    vae_config = create_vae_diffusers_config(original_config, checkpoint=
        checkpoint, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint,
        vae_config)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    config = ClapConfig.from_pretrained('laion/clap-htsat-unfused')
    config.audio_config.update({'patch_embeds_hidden_size': 128,
        'hidden_size': 1024, 'depths': [2, 2, 12, 2]})
    tokenizer = AutoTokenizer.from_pretrained('laion/clap-htsat-unfused')
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        'laion/clap-htsat-unfused')
    converted_text_model = convert_open_clap_checkpoint(checkpoint)
    text_model = ClapModel(config)
    missing_keys, unexpected_keys = text_model.load_state_dict(
        converted_text_model, strict=False)
    missing_keys = list(set(missing_keys) - set(CLAP_EXPECTED_MISSING_KEYS))
    if len(unexpected_keys) > 0:
        raise ValueError(
            f'Unexpected keys when loading CLAP model: {unexpected_keys}')
    if len(missing_keys) > 0:
        raise ValueError(
            f'Missing keys when loading CLAP model: {missing_keys}')
    vocoder_config = create_transformers_vocoder_config(original_config)
    vocoder_config = SpeechT5HifiGanConfig(**vocoder_config)
    converted_vocoder_checkpoint = convert_hifigan_checkpoint(checkpoint,
        vocoder_config)
    vocoder = SpeechT5HifiGan(vocoder_config)
    vocoder.load_state_dict(converted_vocoder_checkpoint)
    pipe = MusicLDMPipeline(vae=vae, text_encoder=text_model, tokenizer=
        tokenizer, unet=unet, scheduler=scheduler, vocoder=vocoder,
        feature_extractor=feature_extractor)
    return pipe
