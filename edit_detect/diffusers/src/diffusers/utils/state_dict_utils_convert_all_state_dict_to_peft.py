def convert_all_state_dict_to_peft(state_dict):
    """
    Attempts to first `convert_state_dict_to_peft`, and if it doesn't detect `lora_linear_layer` for a valid
    `DIFFUSERS` LoRA for example, attempts to exclusively convert the Unet `convert_unet_state_dict_to_peft`
    """
    try:
        peft_dict = convert_state_dict_to_peft(state_dict)
    except Exception as e:
        if str(e) == 'Could not automatically infer state dict type':
            peft_dict = convert_unet_state_dict_to_peft(state_dict)
        else:
            raise
    if not any('lora_A' in key or 'lora_B' in key for key in peft_dict.keys()):
        raise ValueError('Your LoRA was not converted to PEFT')
    return peft_dict
