def save_model(self, output_dir: Optional[str]=None):
    """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.
        """
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    logger.info('Saving model in {}'.format(output_dir))
    if not isinstance(self.model, TFPreTrainedModel):
        raise ValueError('Trainer.model appears to not be a PreTrainedModel')
    self.model.save_pretrained(output_dir)
