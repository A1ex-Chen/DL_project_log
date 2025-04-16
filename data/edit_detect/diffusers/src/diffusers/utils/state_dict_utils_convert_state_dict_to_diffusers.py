def convert_state_dict_to_diffusers(state_dict, original_type=None, **kwargs):
    """
    Converts a state dict to new diffusers format. The state dict can be from previous diffusers format
    (`OLD_DIFFUSERS`), or PEFT format (`PEFT`) or new diffusers format (`DIFFUSERS`). In the last case the method will
    return the state dict as is.

    The method only supports the conversion from diffusers old, PEFT to diffusers new for now.

    Args:
        state_dict (`dict[str, torch.Tensor]`):
            The state dict to convert.
        original_type (`StateDictType`, *optional*):
            The original type of the state dict, if not provided, the method will try to infer it automatically.
        kwargs (`dict`, *args*):
            Additional arguments to pass to the method.

            - **adapter_name**: For example, in case of PEFT, some keys will be pre-pended
                with the adapter name, therefore needs a special handling. By default PEFT also takes care of that in
                `get_peft_model_state_dict` method:
                https://github.com/huggingface/peft/blob/ba0477f2985b1ba311b83459d29895c809404e99/src/peft/utils/save_and_load.py#L92
                but we add it here in case we don't want to rely on that method.
    """
    peft_adapter_name = kwargs.pop('adapter_name', None)
    if peft_adapter_name is not None:
        peft_adapter_name = '.' + peft_adapter_name
    else:
        peft_adapter_name = ''
    if original_type is None:
        if any('to_out_lora' in k for k in state_dict.keys()):
            original_type = StateDictType.DIFFUSERS_OLD
        elif any(f'.lora_A{peft_adapter_name}.weight' in k for k in
            state_dict.keys()):
            original_type = StateDictType.PEFT
        elif any('lora_linear_layer' in k for k in state_dict.keys()):
            return state_dict
        else:
            raise ValueError('Could not automatically infer state dict type')
    if original_type not in DIFFUSERS_STATE_DICT_MAPPINGS.keys():
        raise ValueError(f'Original type {original_type} is not supported')
    mapping = DIFFUSERS_STATE_DICT_MAPPINGS[original_type]
    return convert_state_dict(state_dict, mapping)
