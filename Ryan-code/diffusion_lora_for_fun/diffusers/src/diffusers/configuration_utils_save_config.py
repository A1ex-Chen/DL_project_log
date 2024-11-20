def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub:
    bool=False, **kwargs):
    """
        Save a configuration object to the directory specified in `save_directory` so that it can be reloaded using the
        [`~ConfigMixin.from_config`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file is saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
    if os.path.isfile(save_directory):
        raise AssertionError(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
    os.makedirs(save_directory, exist_ok=True)
    output_config_file = os.path.join(save_directory, self.config_name)
    self.to_json_file(output_config_file)
    logger.info(f'Configuration saved in {output_config_file}')
    if push_to_hub:
        commit_message = kwargs.pop('commit_message', None)
        private = kwargs.pop('private', False)
        create_pr = kwargs.pop('create_pr', False)
        token = kwargs.pop('token', None)
        repo_id = kwargs.pop('repo_id', save_directory.split(os.path.sep)[-1])
        repo_id = create_repo(repo_id, exist_ok=True, private=private,
            token=token).repo_id
        self._upload_folder(save_directory, repo_id, token=token,
            commit_message=commit_message, create_pr=create_pr)
