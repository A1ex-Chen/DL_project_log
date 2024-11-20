def save_pretrained(self, save_directory):
    """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
    assert os.path.isdir(save_directory
        ), 'Saving path should be a directory where the model and configuration can be saved'
    model_to_save = self.module if hasattr(self, 'module') else self
    model_to_save.config.save_pretrained(save_directory)
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info('Model weights saved in {}'.format(output_model_file))
