def test_single_file_components_stage_c(self):
    model_single_file = StableCascadeUNet.from_single_file(
        'https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors'
        , torch_dtype=torch.bfloat16)
    model = StableCascadeUNet.from_pretrained(
        'stabilityai/stable-cascade-prior', variant='bf16', subfolder='prior')
    PARAMS_TO_IGNORE = ['torch_dtype', '_name_or_path',
        '_use_default_values', '_diffusers_version']
    for param_name, param_value in model_single_file.config.items():
        if param_name in PARAMS_TO_IGNORE:
            continue
        assert model.config[param_name
            ] == param_value, f'{param_name} differs between single file loading and pretrained loading'
