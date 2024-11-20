def __init__(self, hparams: argparse.Namespace, num_labels=None, mode=
    'base', config=None, tokenizer=None, model=None, **config_kwargs):
    """Initialize a model, tokenizer and config."""
    super().__init__()
    self.save_hyperparameters(hparams)
    self.step_count = 0
    self.output_dir = Path(self.hparams.output_dir)
    cache_dir = self.hparams.cache_dir if self.hparams.cache_dir else None
    if config is None:
        self.config = AutoConfig.from_pretrained(self.hparams.config_name if
            self.hparams.config_name else self.hparams.model_name_or_path,
            **{'num_labels': num_labels} if num_labels is not None else {},
            cache_dir=cache_dir, **config_kwargs)
    else:
        self.config: PretrainedConfig = config
    extra_model_params = ('encoder_layerdrop', 'decoder_layerdrop',
        'dropout', 'attention_dropout')
    for p in extra_model_params:
        if getattr(self.hparams, p, None):
            assert hasattr(self.config, p
                ), f"model config doesn't have a `{p}` attribute"
            setattr(self.config, p, getattr(self.hparams, p))
    if tokenizer is None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.
            tokenizer_name if self.hparams.tokenizer_name else self.hparams
            .model_name_or_path, cache_dir=cache_dir)
    else:
        self.tokenizer: PreTrainedTokenizer = tokenizer
    self.model_type = MODEL_MODES[mode]
    if model is None:
        self.model = self.model_type.from_pretrained(self.hparams.
            model_name_or_path, from_tf=bool('.ckpt' in self.hparams.
            model_name_or_path), config=self.config, cache_dir=cache_dir)
    else:
        self.model = model
