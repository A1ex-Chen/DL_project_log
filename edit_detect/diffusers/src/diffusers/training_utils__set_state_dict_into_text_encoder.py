def _set_state_dict_into_text_encoder(lora_state_dict: Dict[str, torch.
    Tensor], prefix: str, text_encoder: torch.nn.Module):
    """
    Sets the `lora_state_dict` into `text_encoder` coming from `transformers`.

    Args:
        lora_state_dict: The state dictionary to be set.
        prefix: String identifier to retrieve the portion of the state dict that belongs to `text_encoder`.
        text_encoder: Where the `lora_state_dict` is to be set.
    """
    text_encoder_state_dict = {f"{k.replace(prefix, '')}": v for k, v in
        lora_state_dict.items() if k.startswith(prefix)}
    text_encoder_state_dict = convert_state_dict_to_peft(
        convert_state_dict_to_diffusers(text_encoder_state_dict))
    set_peft_model_state_dict(text_encoder, text_encoder_state_dict,
        adapter_name='default')
