def save_pretrained(self, save_directory: Union[str, os.PathLike],
    is_main_process: bool=True, save_function: Callable=None,
    safe_serialization: bool=True, variant: Optional[str]=None):
    """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~models.adapter.MultiAdapter.from_pretrained`]` class method.

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
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
    idx = 0
    model_path_to_save = save_directory
    for adapter in self.adapters:
        adapter.save_pretrained(model_path_to_save, is_main_process=
            is_main_process, save_function=save_function,
            safe_serialization=safe_serialization, variant=variant)
        idx += 1
        model_path_to_save = model_path_to_save + f'_{idx}'
