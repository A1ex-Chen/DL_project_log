def save_pretrained(self, save_directory):
    """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.
        """
    assert os.path.isdir(save_directory
        ), 'Saving path should be a directory where the model and configuration can be saved'
    output_config_file = os.path.join(save_directory, CONFIG_NAME)
    self.to_json_file(output_config_file)
    logger.info('Configuration saved in {}'.format(output_config_file))
