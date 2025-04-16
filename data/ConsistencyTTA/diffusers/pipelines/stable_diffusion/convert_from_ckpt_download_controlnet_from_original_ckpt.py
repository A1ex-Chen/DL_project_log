def download_controlnet_from_original_ckpt(checkpoint_path: str,
    original_config_file: str, image_size: int=512, extract_ema: bool=False,
    num_in_channels: Optional[int]=None, upcast_attention: Optional[bool]=
    None, device: str=None, from_safetensors: bool=False
    ) ->StableDiffusionPipeline:
    if not is_omegaconf_available():
        raise ValueError(BACKENDS_MAPPING['omegaconf'][1])
    from omegaconf import OmegaConf
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
    while 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    original_config = OmegaConf.load(original_config_file)
    if num_in_channels is not None:
        original_config['model']['params']['unet_config']['params'][
            'in_channels'] = num_in_channels
    if 'control_stage_config' not in original_config.model.params:
        raise ValueError(
            '`control_stage_config` not present in original config')
    controlnet_model = convert_controlnet_checkpoint(checkpoint,
        original_config, checkpoint_path, image_size, upcast_attention,
        extract_ema)
    return controlnet_model
