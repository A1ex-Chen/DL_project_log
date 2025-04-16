def save_pretrained(self, save_directory):
    """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        :func:`~transformers.TFPreTrainedModel.from_pretrained` class method.

        Arguments:
            save_directory (:obj:`str`):
                Directory to which to save. Will be created if it doesn't exist.
        """
    if os.path.isfile(save_directory):
        logger.error('Provided path ({}) should be a directory, not a file'
            .format(save_directory))
        return
    os.makedirs(save_directory, exist_ok=True)
    self.config.save_pretrained(save_directory)
    output_model_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)
    self.save_weights(output_model_file)
    logger.info('Model weights saved in {}'.format(output_model_file))
