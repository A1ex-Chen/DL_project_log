def text_encoder_lora_state_dict(text_encoder):
    deprecate('text_encoder_load_state_dict in `models`', '0.27.0',
        '`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.'
        )
    state_dict = {}
    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f'{name}.q_proj.lora_linear_layer.{k}'] = v
        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f'{name}.k_proj.lora_linear_layer.{k}'] = v
        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f'{name}.v_proj.lora_linear_layer.{k}'] = v
        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f'{name}.out_proj.lora_linear_layer.{k}'] = v
    return state_dict
