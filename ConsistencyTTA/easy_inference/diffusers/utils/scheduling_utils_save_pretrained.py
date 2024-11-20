def save_pretrained(self, save_directory: Union[str, os.PathLike],
    push_to_hub: bool=False, **kwargs):
    """
        Save a scheduler configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~SchedulerMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
    self.save_config(save_directory=save_directory, push_to_hub=push_to_hub,
        **kwargs)
