def save_pretrained(self, save_directory: Union[str, os.PathLike]):
    """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
        """
    if os.path.isfile(save_directory):
        raise AssertionError(
            'Provided path ({}) should be a directory, not a file'.format(
            save_directory))
    os.makedirs(save_directory, exist_ok=True)
    output_config_file = os.path.join(save_directory, CONFIG_NAME)
    self.to_json_file(output_config_file, use_diff=True)
    logger.info('Configuration saved in {}'.format(output_config_file))
