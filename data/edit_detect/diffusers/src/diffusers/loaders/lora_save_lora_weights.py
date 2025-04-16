@classmethod
def save_lora_weights(cls, save_directory: Union[str, os.PathLike],
    unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]]=None,
    text_encoder_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor
    ]]=None, text_encoder_2_lora_layers: Dict[str, Union[torch.nn.Module,
    torch.Tensor]]=None, is_main_process: bool=True, weight_name: str=None,
    save_function: Callable=None, safe_serialization: bool=True):
    """
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `unet`.
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
            text_encoder_2_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder_2`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        """
    state_dict = {}

    def pack_weights(layers, prefix):
        layers_weights = layers.state_dict() if isinstance(layers, torch.nn
            .Module) else layers
        layers_state_dict = {f'{prefix}.{module_name}': param for 
            module_name, param in layers_weights.items()}
        return layers_state_dict
    if not (unet_lora_layers or text_encoder_lora_layers or
        text_encoder_2_lora_layers):
        raise ValueError(
            'You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`.'
            )
    if unet_lora_layers:
        state_dict.update(pack_weights(unet_lora_layers, 'unet'))
    if text_encoder_lora_layers:
        state_dict.update(pack_weights(text_encoder_lora_layers,
            'text_encoder'))
    if text_encoder_2_lora_layers:
        state_dict.update(pack_weights(text_encoder_2_lora_layers,
            'text_encoder_2'))
    cls.write_lora_layers(state_dict=state_dict, save_directory=
        save_directory, is_main_process=is_main_process, weight_name=
        weight_name, save_function=save_function, safe_serialization=
        safe_serialization)
