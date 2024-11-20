@classmethod
def save_lora_weights(self, save_directory: Union[str, os.PathLike],
    unet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]]=None,
    text_encoder_lora_layers: Dict[str, torch.nn.Module]=None,
    is_main_process: bool=True, weight_name: str=None, save_function:
    Callable=None, safe_serialization: bool=False):
    """
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the UNet.
            text_encoder_lora_layers (`Dict[str, torch.nn.Module] or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes 🤗 Transformers.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
    if os.path.isfile(save_directory):
        logger.error(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
        return
    if save_function is None:
        if safe_serialization:

            def save_function(weights, filename):
                return safetensors.torch.save_file(weights, filename,
                    metadata={'format': 'pt'})
        else:
            save_function = torch.save
    os.makedirs(save_directory, exist_ok=True)
    state_dict = {}
    if unet_lora_layers is not None:
        weights = unet_lora_layers.state_dict() if isinstance(unet_lora_layers,
            torch.nn.Module) else unet_lora_layers
        unet_lora_state_dict = {f'{self.unet_name}.{module_name}': param for
            module_name, param in weights.items()}
        state_dict.update(unet_lora_state_dict)
    if text_encoder_lora_layers is not None:
        weights = text_encoder_lora_layers.state_dict() if isinstance(
            text_encoder_lora_layers, torch.nn.Module
            ) else text_encoder_lora_layers
        text_encoder_lora_state_dict = {
            f'{self.text_encoder_name}.{module_name}': param for 
            module_name, param in weights.items()}
        state_dict.update(text_encoder_lora_state_dict)
    if weight_name is None:
        if safe_serialization:
            weight_name = LORA_WEIGHT_NAME_SAFE
        else:
            weight_name = LORA_WEIGHT_NAME
    save_function(state_dict, os.path.join(save_directory, weight_name))
    logger.info(
        f'Model weights saved in {os.path.join(save_directory, weight_name)}')
