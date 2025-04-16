def _save(self, output_dir: Optional[str]=None):
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info('Saving model checkpoint to %s', output_dir)
    if not isinstance(self.model, PreTrainedModel):
        logger.info(
            'Trainer.model is not a `PreTrainedModel`, only saving its state dict.'
            )
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        self.model.save_pretrained(output_dir)
    if self.tokenizer is not None and self.is_world_process_zero():
        self.tokenizer.save_pretrained(output_dir)
    torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
