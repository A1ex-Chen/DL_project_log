def convert_gligen_to_diffusers(checkpoint_path: str, original_config_file:
    str, attention_type: str, image_size: int=512, extract_ema: bool=False,
    num_in_channels: int=None, device: str=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'global_step' in checkpoint:
        checkpoint['global_step']
    else:
        print('global_step key not found in model')
    original_config = yaml.safe_load(original_config_file)
    if num_in_channels is not None:
        original_config['model']['params']['in_channels'] = num_in_channels
    num_train_timesteps = original_config['diffusion']['params']['timesteps']
    beta_start = original_config['diffusion']['params']['linear_start']
    beta_end = original_config['diffusion']['params']['linear_end']
    scheduler = DDIMScheduler(beta_end=beta_end, beta_schedule=
        'scaled_linear', beta_start=beta_start, num_train_timesteps=
        num_train_timesteps, steps_offset=1, clip_sample=False,
        set_alpha_to_one=False, prediction_type='epsilon')
    unet_config = create_unet_config(original_config, image_size,
        attention_type)
    unet = UNet2DConditionModel(**unet_config)
    converted_unet_checkpoint = convert_gligen_unet_checkpoint(checkpoint,
        unet_config, path=checkpoint_path, extract_ema=extract_ema)
    unet.load_state_dict(converted_unet_checkpoint)
    vae_config = create_vae_config(original_config, image_size)
    converted_vae_checkpoint = convert_gligen_vae_checkpoint(checkpoint,
        vae_config)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    text_encoder = convert_open_clip_checkpoint(checkpoint)
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    if attention_type == 'gated-text-image':
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            'openai/clip-vit-large-patch14')
        processor = CLIPProcessor.from_pretrained(
            'openai/clip-vit-large-patch14')
        pipe = StableDiffusionGLIGENTextImagePipeline(vae=vae, text_encoder
            =text_encoder, tokenizer=tokenizer, image_encoder=image_encoder,
            processor=processor, unet=unet, scheduler=scheduler,
            safety_checker=None, feature_extractor=None)
    elif attention_type == 'gated':
        pipe = StableDiffusionGLIGENPipeline(vae=vae, text_encoder=
            text_encoder, tokenizer=tokenizer, unet=unet, scheduler=
            scheduler, safety_checker=None, feature_extractor=None)
    return pipe
