def get_module_kohya_state_dict(module, prefix: str, dtype: torch.dtype,
    adapter_name: str='default'):
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=
        adapter_name).items():
        kohya_key = peft_key.replace('base_model.model', prefix)
        kohya_key = kohya_key.replace('lora_A', 'lora_down')
        kohya_key = kohya_key.replace('lora_B', 'lora_up')
        kohya_key = kohya_key.replace('.', '_', kohya_key.count('.') - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        if 'lora_down' in kohya_key:
            alpha_key = f"{kohya_key.split('.')[0]}.alpha"
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.
                peft_config[adapter_name].lora_alpha).to(dtype)
    return kohya_ss_state_dict
