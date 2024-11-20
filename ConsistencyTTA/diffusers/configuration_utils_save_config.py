def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub:
    bool=False, **kwargs):
    """
        Save a configuration object to the directory specified in `save_directory` so that it can be reloaded using the
        [`~ConfigMixin.from_config`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file is saved (will be created if it does not exist).
        """
    if os.path.isfile(save_directory):
        raise AssertionError(
            f'Provided path ({save_directory}) should be a directory, not a file'
            )
    os.makedirs(save_directory, exist_ok=True)
    output_config_file = os.path.join(save_directory, self.config_name)
    self.to_json_file(output_config_file)
    logger.info(f'Configuration saved in {output_config_file}')
