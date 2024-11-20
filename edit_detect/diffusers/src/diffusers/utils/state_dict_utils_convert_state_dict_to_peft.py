def convert_state_dict_to_peft(state_dict, original_type=None, **kwargs):
    """
    Converts a state dict to the PEFT format The state dict can be from previous diffusers format (`OLD_DIFFUSERS`), or
    new diffusers format (`DIFFUSERS`). The method only supports the conversion from diffusers old/new to PEFT for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
    """
    if original_type is None:
        if any('to_out_lora' in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        elif any('lora_linear_layer' in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS
        else:
            raise ValueError('Could not automatically infer state dict type')
    if original_type not in PEFT_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f'Original type {original_type} is not supported')
    mapping = PEFT_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)
