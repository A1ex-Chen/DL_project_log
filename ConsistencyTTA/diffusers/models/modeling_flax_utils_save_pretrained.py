def save_pretrained(self, save_directory: Union[str, os.PathLike], params:
    Union[Dict, FrozenDict], is_main_process: bool=True):
    """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~FlaxModelMixin.from_pretrained`]` class method

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
        """
    if os.path.isfile(save_directory):
        logger.error(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
        return
    os.makedirs(save_directory, exist_ok=True)
    model_to_save = self
    if is_main_process:
        model_to_save.save_config(save_directory)
    output_model_file = os.path.join(save_directory, FLAX_WEIGHTS_NAME)
    with open(output_model_file, 'wb') as f:
        model_bytes = to_bytes(params)
        f.write(model_bytes)
    logger.info(f'Model weights saved in {output_model_file}')
