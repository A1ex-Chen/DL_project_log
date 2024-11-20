def save_pretrained(self, save_directory: Union[str, os.PathLike], params:
    Union[Dict, FrozenDict], is_main_process: bool=True, push_to_hub: bool=
    False, **kwargs):
    """
        Save a model and its configuration file to a directory so that it can be reloaded using the
        [`~FlaxModelMixin.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a model and its configuration file to. Will be created if it doesn't exist.
            params (`Union[Dict, FrozenDict]`):
                A `PyTree` of model parameters.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
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
    output_model_file = os.path.join(save_directory, FLAX_WEIGHTS_NAME)
    with open(output_model_file, 'wb') as f:
        model_bytes = to_bytes(params)
        f.write(model_bytes)
    logger.info(f'Model weights saved in {output_model_file}')
    if push_to_hub:
        self._upload_folder(save_directory, repo_id, token=token,
            commit_message=commit_message, create_pr=create_pr)
