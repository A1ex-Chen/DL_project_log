def get_super_res_unet(unet_checkpoint_path, verify_param_count=True,
    sample_size=None):
    orig_path = unet_checkpoint_path
    original_unet_config = yaml.safe_load(os.path.join(orig_path, 'config.yml')
        )
    original_unet_config = original_unet_config['params']
    unet_diffusers_config = superres_create_unet_diffusers_config(
        original_unet_config)
    unet_diffusers_config['time_embedding_dim'] = original_unet_config[
        'model_channels'] * int(original_unet_config['channel_mult'].split(
        ',')[-1])
    if original_unet_config['encoder_dim'] != original_unet_config[
        'encoder_channels']:
        unet_diffusers_config['encoder_hid_dim'] = original_unet_config[
            'encoder_dim']
        unet_diffusers_config['class_embed_type'] = 'timestep'
        unet_diffusers_config['addition_embed_type'] = 'text'
    unet_diffusers_config['time_embedding_act_fn'] = 'gelu'
    unet_diffusers_config['resnet_skip_time_act'] = True
    unet_diffusers_config['resnet_out_scale_factor'] = 1 / 0.7071
    unet_diffusers_config['mid_block_scale_factor'] = 1 / 0.7071
    unet_diffusers_config['only_cross_attention'] = bool(original_unet_config
        ['disable_self_attentions']
        ) if 'disable_self_attentions' in original_unet_config and isinstance(
        original_unet_config['disable_self_attentions'], int) else True
    if sample_size is None:
        unet_diffusers_config['sample_size'] = original_unet_config[
            'image_size']
    else:
        unet_diffusers_config['sample_size'] = sample_size
    unet_checkpoint = torch.load(os.path.join(unet_checkpoint_path,
        'pytorch_model.bin'), map_location='cpu')
    if verify_param_count:
        verify_param_count(orig_path, unet_diffusers_config)
    converted_unet_checkpoint = superres_convert_ldm_unet_checkpoint(
        unet_checkpoint, unet_diffusers_config, path=unet_checkpoint_path)
    converted_keys = converted_unet_checkpoint.keys()
    model = UNet2DConditionModel(**unet_diffusers_config)
    expected_weights = model.state_dict().keys()
    diff_c_e = set(converted_keys) - set(expected_weights)
    diff_e_c = set(expected_weights) - set(converted_keys)
    assert len(diff_e_c) == 0, f'Expected, but not converted: {diff_e_c}'
    assert len(diff_c_e) == 0, f'Converted, but not expected: {diff_c_e}'
    model.load_state_dict(converted_unet_checkpoint)
    return model
