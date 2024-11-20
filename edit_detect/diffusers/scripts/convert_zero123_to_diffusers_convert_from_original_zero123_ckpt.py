def convert_from_original_zero123_ckpt(checkpoint_path,
    original_config_file, extract_ema, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt['global_step']
    checkpoint = ckpt['state_dict']
    del ckpt
    torch.cuda.empty_cache()
    original_config = yaml.safe_load(original_config_file)
    original_config['model']['params']['cond_stage_config']['target'].split('.'
        )[-1]
    num_in_channels = 8
    original_config['model']['params']['unet_config']['params']['in_channels'
        ] = num_in_channels
    prediction_type = 'epsilon'
    image_size = 256
    num_train_timesteps = getattr(original_config['model']['params'],
        'timesteps', None) or 1000
    beta_start = getattr(original_config['model']['params'], 'linear_start',
        None) or 0.02
    beta_end = getattr(original_config['model']['params'], 'linear_end', None
        ) or 0.085
    scheduler = DDIMScheduler(beta_end=beta_end, beta_schedule=
        'scaled_linear', beta_start=beta_start, num_train_timesteps=
        num_train_timesteps, steps_offset=1, clip_sample=False,
        set_alpha_to_one=False, prediction_type=prediction_type)
    scheduler.register_to_config(clip_sample=False)
    upcast_attention = None
    unet_config = create_unet_diffusers_config(original_config, image_size=
        image_size)
    unet_config['upcast_attention'] = upcast_attention
    with init_empty_weights():
        unet = UNet2DConditionModel(**unet_config)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(checkpoint,
        unet_config, path=None, extract_ema=extract_ema)
    for param_name, param in converted_unet_checkpoint.items():
        set_module_tensor_to_device(unet, param_name, 'cpu', value=param)
    vae_config = create_vae_diffusers_config(original_config, image_size=
        image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint,
        vae_config)
    if 'model' in original_config and 'params' in original_config['model'
        ] and 'scale_factor' in original_config['model']['params']:
        vae_scaling_factor = original_config['model']['params']['scale_factor']
    else:
        vae_scaling_factor = 0.18215
    vae_config['scaling_factor'] = vae_scaling_factor
    with init_empty_weights():
        vae = AutoencoderKL(**vae_config)
    for param_name, param in converted_vae_checkpoint.items():
        set_module_tensor_to_device(vae, param_name, 'cpu', value=param)
    feature_extractor = CLIPImageProcessor.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', subfolder=
        'feature_extractor')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', subfolder='image_encoder')
    cc_projection = CCProjection()
    cc_projection.load_state_dict({'projection.weight': checkpoint[
        'cc_projection.weight'].cpu(), 'projection.bias': checkpoint[
        'cc_projection.bias'].cpu()})
    pipe = Zero1to3StableDiffusionPipeline(vae, image_encoder, unet,
        scheduler, None, feature_extractor, cc_projection,
        requires_safety_checker=False)
    return pipe
