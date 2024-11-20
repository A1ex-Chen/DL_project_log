def save_pretrained(self, save_directory: Union[str, os.PathLike],
    push_to_hub: bool=False, **kwargs):
    """
        Save a scheduler configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~FlaxSchedulerMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
    self.save_config(save_directory=save_directory, push_to_hub=push_to_hub,
        **kwargs)
