def convert_controlnet_checkpoint(checkpoint, original_config,
    checkpoint_path, image_size, upcast_attention, extract_ema,
    use_linear_projection=None, cross_attention_dim=None):
    ctrlnet_config = create_unet_diffusers_config(original_config,
        image_size=image_size, controlnet=True)
    ctrlnet_config['upcast_attention'] = upcast_attention
    ctrlnet_config.pop('sample_size')
    if use_linear_projection is not None:
        ctrlnet_config['use_linear_projection'] = use_linear_projection
    if cross_attention_dim is not None:
        ctrlnet_config['cross_attention_dim'] = cross_attention_dim
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        controlnet = ControlNetModel(**ctrlnet_config)
    if 'time_embed.0.weight' in checkpoint:
        skip_extract_state_dict = True
    else:
        skip_extract_state_dict = False
    converted_ctrl_checkpoint = convert_ldm_unet_checkpoint(checkpoint,
        ctrlnet_config, path=checkpoint_path, extract_ema=extract_ema,
        controlnet=True, skip_extract_state_dict=skip_extract_state_dict)
    if is_accelerate_available():
        for param_name, param in converted_ctrl_checkpoint.items():
            set_module_tensor_to_device(controlnet, param_name, 'cpu',
                value=param)
    else:
        controlnet.load_state_dict(converted_ctrl_checkpoint)
    return controlnet
