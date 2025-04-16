def save_pretrained(self, save_directory: Union[str, os.PathLike],
    is_main_process: bool=True, save_function: Callable=None,
    safe_serialization: bool=False, variant: Optional[str]=None):
    """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~models.ModelMixin.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
    if safe_serialization and not is_safetensors_available():
        raise ImportError(
            '`safe_serialization` requires the `safetensors library: `pip install safetensors`.'
            )
    if os.path.isfile(save_directory):
        logger.error(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
        return
    os.makedirs(save_directory, exist_ok=True)
    model_to_save = self
    if is_main_process:
        model_to_save.save_config(save_directory)
    state_dict = model_to_save.state_dict()
    weights_name = (SAFETENSORS_WEIGHTS_NAME if safe_serialization else
        WEIGHTS_NAME)
    weights_name = _add_variant(weights_name, variant)
    if safe_serialization:
        safetensors.torch.save_file(state_dict, os.path.join(save_directory,
            weights_name), metadata={'format': 'pt'})
    else:
        torch.save(state_dict, os.path.join(save_directory, weights_name))
    logger.info(
        f'Model weights saved in {os.path.join(save_directory, weights_name)}')
