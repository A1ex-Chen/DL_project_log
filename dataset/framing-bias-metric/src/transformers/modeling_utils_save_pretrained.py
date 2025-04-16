def save_pretrained(self, save_directory: Union[str, os.PathLike]):
    """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
    if os.path.isfile(save_directory):
        logger.error('Provided path ({}) should be a directory, not a file'
            .format(save_directory))
        return
    os.makedirs(save_directory, exist_ok=True)
    model_to_save = self.module if hasattr(self, 'module') else self
    model_to_save.config.architectures = [model_to_save.__class__.__name__]
    state_dict = model_to_save.state_dict()
    if self._keys_to_ignore_on_save is not None:
        state_dict = {k: v for k, v in state_dict.items() if k not in self.
            _keys_to_ignore_on_save}
    output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
    if getattr(self.config, 'xla_device', False) and is_torch_tpu_available():
        import torch_xla.core.xla_model as xm
        if xm.is_master_ordinal():
            model_to_save.config.save_pretrained(save_directory)
        xm.save(state_dict, output_model_file)
    else:
        model_to_save.config.save_pretrained(save_directory)
        torch.save(state_dict, output_model_file)
    logger.info('Model weights saved in {}'.format(output_model_file))
