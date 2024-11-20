def __init__(self, config=None, data_args=None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if config is None:
        assert isinstance(self.model, PreTrainedModel
            ), f'If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is {self.model.__class__}'
        self.config = self._actual_model(self.model).config
    else:
        self.config = config
    self.data_args = data_args
    self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config,
        FSMTConfig) else self.config.vocab_size
    if (self.args.label_smoothing != 0 or self.data_args is not None and
        self.data_args.ignore_pad_token_for_loss):
        assert self.config.pad_token_id is not None, 'Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss calculation or doing label smoothing.'
    if (self.config.pad_token_id is None and self.config.eos_token_id is not
        None):
        logger.warn(
            f'The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for padding..'
            )
    if self.args.label_smoothing == 0:
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.
            pad_token_id)
    else:
        from utils import label_smoothed_nll_loss
        self.loss_fn = label_smoothed_nll_loss
