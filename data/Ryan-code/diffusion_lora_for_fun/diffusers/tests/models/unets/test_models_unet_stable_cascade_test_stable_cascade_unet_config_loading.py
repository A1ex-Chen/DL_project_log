def test_stable_cascade_unet_config_loading(self):
    config = StableCascadeUNet.load_config(pretrained_model_name_or_path=
        'diffusers/stable-cascade-configs', subfolder='prior')
    single_file_url = (
        'https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors'
        )
    single_file_unet = StableCascadeUNet.from_single_file(single_file_url,
        config=config)
    single_file_unet_config = single_file_unet.config
    del single_file_unet
    gc.collect()
    torch.cuda.empty_cache()
    PARAMS_TO_IGNORE = ['torch_dtype', '_name_or_path',
        '_use_default_values', '_diffusers_version']
    for param_name, param_value in config.items():
        if param_name in PARAMS_TO_IGNORE:
            continue
        assert single_file_unet_config[param_name] == param_value
