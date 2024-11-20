def save_attn_procs(self, save_directory: Union[str, os.PathLike],
    is_main_process: bool=True, weight_name: str=None, save_function:
    Callable=None, safe_serialization: bool=False, **kwargs):
    """
        Save an attention processor to a directory so that it can be reloaded using the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.

        """
    weight_name = weight_name or deprecate('weights_name', '0.20.0',
        '`weights_name` is deprecated, please use `weight_name` instead.',
        take_from=kwargs)
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
    is_custom_diffusion = any(isinstance(x, (CustomDiffusionAttnProcessor,
        CustomDiffusionXFormersAttnProcessor)) for _, x in self.
        attn_processors.items())
    if is_custom_diffusion:
        model_to_save = AttnProcsLayers({y: x for y, x in self.
            attn_processors.items() if isinstance(x, (
            CustomDiffusionAttnProcessor,
            CustomDiffusionXFormersAttnProcessor))})
        state_dict = model_to_save.state_dict()
        for name, attn in self.attn_processors.items():
            if len(attn.state_dict()) == 0:
                state_dict[name] = {}
    else:
        model_to_save = AttnProcsLayers(self.attn_processors)
        state_dict = model_to_save.state_dict()
    if weight_name is None:
        if safe_serialization:
            weight_name = (CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if
                is_custom_diffusion else LORA_WEIGHT_NAME_SAFE)
        else:
            weight_name = (CUSTOM_DIFFUSION_WEIGHT_NAME if
                is_custom_diffusion else LORA_WEIGHT_NAME)
    save_function(state_dict, os.path.join(save_directory, weight_name))
    logger.info(
        f'Model weights saved in {os.path.join(save_directory, weight_name)}')
