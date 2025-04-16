def test_stable_cascade_unet_prior_single_file_components(self):
    single_file_url = (
        'https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors'
        )
    single_file_unet = StableCascadeUNet.from_single_file(single_file_url)
    single_file_unet_config = single_file_unet.config
    del single_file_unet
    gc.collect()
    torch.cuda.empty_cache()
    unet = StableCascadeUNet.from_pretrained('stabilityai/stable-cascade-prior'
        , subfolder='prior', variant='bf16')
    unet_config = unet.config
    del unet
    gc.collect()
    torch.cuda.empty_cache()
    PARAMS_TO_IGNORE = ['torch_dtype', '_name_or_path',
        '_use_default_values', '_diffusers_version']
    for param_name, param_value in single_file_unet_config.items():
        if param_name in PARAMS_TO_IGNORE:
            continue
        assert unet_config[param_name] == param_value
