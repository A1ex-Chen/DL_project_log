def load_pipeline_from_original_AudioLDM2_ckpt(checkpoint_path: str,
    original_config_file: str=None, image_size: int=1024, prediction_type:
    str=None, extract_ema: bool=False, scheduler_type: str='ddim',
    cross_attention_dim: Union[List, List[List]]=None,
    transformer_layers_per_block: int=None, device: str=None,
    from_safetensors: bool=False) ->AudioLDM2Pipeline:
    """
    Load an AudioLDM2 pipeline object from a `.ckpt`/`.safetensors` file and (ideally) a `.yaml` config file.

    Although many of the arguments can be automatically inferred, some of these rely on brittle checks against the
    global step count, which will likely fail for models that have undergone further fine-tuning. Therefore, it is
    recommended that you override the default values and/or supply an `original_config_file` wherever possible.

    Args:
        checkpoint_path (`str`): Path to `.ckpt` file.
        original_config_file (`str`):
            Path to `.yaml` config file corresponding to the original architecture. If `None`, will be automatically
            set to the AudioLDM2 base config.
        image_size (`int`, *optional*, defaults to 1024):
            The image size that the model was trained on.
        prediction_type (`str`, *optional*):
            The prediction type that the model was trained on. If `None`, will be automatically
            inferred by looking for a key in the config. For the default config, the prediction type is `'epsilon'`.
        scheduler_type (`str`, *optional*, defaults to 'ddim'):
            Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
            "ddim"]`.
        cross_attention_dim (`list`, *optional*, defaults to `None`):
            The dimension of the cross-attention layers. If `None`, the cross-attention dimension will be
            automatically inferred. Set to `[768, 1024]` for the base model, or `[768, 1024, None]` for the large model.
        transformer_layers_per_block (`int`, *optional*, defaults to `None`):
            The number of transformer layers in each transformer block. If `None`, number of layers will be "
             "automatically inferred. Set to `1` for the base model, or `2` for the large model.
        extract_ema (`bool`, *optional*, defaults to `False`): Only relevant for
            checkpoints that have both EMA and non-EMA weights. Whether to extract the EMA weights or not. Defaults to
            `False`. Pass `True` to extract the EMA weights. EMA weights usually yield higher quality images for
            inference. Non-EMA weights are usually better to continue fine-tuning.
        device (`str`, *optional*, defaults to `None`):
            The device to use. Pass `None` to determine automatically.
        from_safetensors (`str`, *optional*, defaults to `False`):
            If `checkpoint_path` is in `safetensors` format, load checkpoint with safetensors instead of PyTorch.
        return: An AudioLDM2Pipeline object representing the passed-in `.ckpt`/`.safetensors` file.
    """
    if from_safetensors:
        if not is_safetensors_available():
            raise ValueError(BACKENDS_MAPPING['safetensors'][1])
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
    if image_size is not None:
        original_config['model']['params']['unet_config']['params'][
            'image_size'] = image_size
    if cross_attention_dim is not None:
        original_config['model']['params']['unet_config']['params'][
            'context_dim'] = cross_attention_dim
    if transformer_layers_per_block is not None:
        original_config['model']['params']['unet_config']['params'][
            'transformer_depth'] = transformer_layers_per_block
    if 'parameterization' in original_config['model']['params'
        ] and original_config['model']['params']['parameterization'] == 'v':
        if prediction_type is None:
            prediction_type = 'v_prediction'
    elif prediction_type is None:
        prediction_type = 'epsilon'
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
    unet = AudioLDM2UNet2DConditionModel(**unet_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint,
        unet_config, path=checkpoint_path, extract_ema=extract_ema)
    unet.load_state_dict(converted_unet_checkpoint)
    vae_config = create_vae_diffusers_config(original_config, checkpoint=
        checkpoint, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint,
        vae_config)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    clap_config = ClapConfig.from_pretrained('laion/clap-htsat-unfused')
    clap_config.audio_config.update({'patch_embeds_hidden_size': 128,
        'hidden_size': 1024, 'depths': [2, 2, 12, 2]})
    clap_tokenizer = AutoTokenizer.from_pretrained('laion/clap-htsat-unfused')
    clap_feature_extractor = AutoFeatureExtractor.from_pretrained(
        'laion/clap-htsat-unfused')
    converted_clap_model = convert_open_clap_checkpoint(checkpoint)
    clap_model = ClapModel(clap_config)
    missing_keys, unexpected_keys = clap_model.load_state_dict(
        converted_clap_model, strict=False)
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
    t5_config = T5Config.from_pretrained('google/flan-t5-large')
    converted_t5_checkpoint = extract_sub_model(checkpoint, key_prefix=
        'cond_stage_models.1.model.')
    t5_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
    t5_tokenizer.model_max_length = 128
    t5_model = T5EncoderModel(t5_config)
    t5_model.load_state_dict(converted_t5_checkpoint)
    gpt2_config = GPT2Config.from_pretrained('gpt2')
    gpt2_model = GPT2Model(gpt2_config)
    gpt2_model.config.max_new_tokens = original_config['model']['params'][
        'cond_stage_config']['crossattn_audiomae_generated']['params'][
        'sequence_gen_length']
    converted_gpt2_checkpoint = extract_sub_model(checkpoint, key_prefix=
        'cond_stage_models.0.model.')
    gpt2_model.load_state_dict(converted_gpt2_checkpoint)
    projection_model = AudioLDM2ProjectionModel(clap_config.projection_dim,
        t5_config.d_model, gpt2_config.n_embd)
    converted_projection_checkpoint = convert_projection_checkpoint(checkpoint)
    projection_model.load_state_dict(converted_projection_checkpoint)
    pipe = AudioLDM2Pipeline(vae=vae, text_encoder=clap_model,
        text_encoder_2=t5_model, projection_model=projection_model,
        language_model=gpt2_model, tokenizer=clap_tokenizer, tokenizer_2=
        t5_tokenizer, feature_extractor=clap_feature_extractor, unet=unet,
        scheduler=scheduler, vocoder=vocoder)
    return pipe
