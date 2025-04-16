def convert_controlnet_checkpoint(checkpoint, original_config,
    checkpoint_path, image_size, upcast_attention, extract_ema):
    ctrlnet_config = create_unet_diffusers_config(original_config,
        image_size=image_size, controlnet=True)
    ctrlnet_config['upcast_attention'] = upcast_attention
    ctrlnet_config.pop('sample_size')
    controlnet_model = ControlNetModel(**ctrlnet_config)
    converted_ctrl_checkpoint = convert_ldm_unet_checkpoint(checkpoint,
        ctrlnet_config, path=checkpoint_path, extract_ema=extract_ema,
        controlnet=True)
    controlnet_model.load_state_dict(converted_ctrl_checkpoint)
    return controlnet_model
