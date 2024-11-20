def save_pretrained(self, save_directory):
    """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
    assert os.path.isdir(save_directory
        ), 'Saving path should be a directory where the model and configuration can be saved'
    self.config.save_pretrained(save_directory)
    output_model_file = os.path.join(save_directory, TF2_WEIGHTS_NAME)
    self.save_weights(output_model_file)
    logger.info('Model weights saved in {}'.format(output_model_file))
