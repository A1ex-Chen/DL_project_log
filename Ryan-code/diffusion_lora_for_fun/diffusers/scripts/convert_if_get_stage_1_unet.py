def get_stage_1_unet(unet_config, unet_checkpoint_path):
    original_unet_config = yaml.safe_load(unet_config)
    original_unet_config = original_unet_config['params']
    unet_diffusers_config = create_unet_diffusers_config(original_unet_config)
    unet = UNet2DConditionModel(**unet_diffusers_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet_checkpoint = torch.load(unet_checkpoint_path, map_location=device)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(unet_checkpoint,
        unet_diffusers_config, path=unet_checkpoint_path)
    unet.load_state_dict(converted_unet_checkpoint)
    return unet
