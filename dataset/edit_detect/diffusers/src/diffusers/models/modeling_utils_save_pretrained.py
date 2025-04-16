def save_pretrained(self, save_directory: Union[str, os.PathLike],
    is_main_process: bool=True, save_function: Optional[Callable]=None,
    safe_serialization: bool=True, variant: Optional[str]=None, push_to_hub:
    bool=False, **kwargs):
    """
        Save a model and its configuration file to a directory so that it can be reloaded using the
        [`~models.ModelMixin.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a model and its configuration file to. Will be created if it doesn't exist.
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
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
    if os.path.isfile(save_directory):
        logger.error(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
        return
    os.makedirs(save_directory, exist_ok=True)
    if push_to_hub:
        commit_message = kwargs.pop('commit_message', None)
        private = kwargs.pop('private', False)
        create_pr = kwargs.pop('create_pr', False)
        token = kwargs.pop('token', None)
        repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
        repo_id = create_repo(repo_id, exist_ok=True, private=private,
            token=token).repo_id
    model_to_save = self
    if is_main_process:
        model_to_save.save_config(save_directory)
    state_dict = model_to_save.state_dict()
    weights_name = (SAFETENSORS_WEIGHTS_NAME if safe_serialization else
        WEIGHTS_NAME)
    weights_name = _add_variant(weights_name, variant)
    if safe_serialization:
        safetensors.torch.save_file(state_dict, Path(save_directory,
            weights_name).as_posix(), metadata={'format': 'pt'})
    else:
        torch.save(state_dict, Path(save_directory, weights_name).as_posix())
    logger.info(
        f'Model weights saved in {Path(save_directory, weights_name).as_posix()}'
        )
    if push_to_hub:
        model_card = load_or_create_model_card(repo_id, token=token)
        model_card = populate_model_card(model_card)
        model_card.save(Path(save_directory, 'README.md').as_posix())
        self._upload_folder(save_directory, repo_id, token=token,
            commit_message=commit_message, create_pr=create_pr)
