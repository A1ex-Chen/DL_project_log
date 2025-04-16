def __init__(self, hparams, **kwargs):
    if hparams.sortish_sampler and hparams.gpus > 1:
        hparams.replace_sampler_ddp = False
    elif hparams.max_tokens_per_batch is not None:
        if hparams.gpus > 1:
            raise NotImplementedError(
                'Dynamic Batch size does not work for multi-gpu training')
        if hparams.sortish_sampler:
            raise ValueError(
                '--sortish_sampler and --max_tokens_per_batch may not be used simultaneously'
                )
    super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
    use_task_specific_params(self.model, 'summarization')
    self.metrics_save_path = Path(self.output_dir) / 'metrics.json'
    self.hparams_save_path = Path(self.output_dir) / 'hparams.pkl'
    pickle_save(self.hparams, self.hparams_save_path)
    self.step_count = 0
    self.metrics = defaultdict(list)
    self.model_type = self.config.model_type
    self.vocab_size = (self.config.tgt_vocab_size if self.model_type ==
        'fsmt' else self.config.vocab_size)
    self.dataset_kwargs: dict = dict(data_dir=self.hparams.data_dir,
        max_source_length=self.hparams.max_source_length, prefix=self.model
        .config.prefix or '')
    n_observations_per_split = {'train': self.hparams.n_train, 'val': self.
        hparams.n_val, 'test': self.hparams.n_test}
    self.n_obs = {k: (v if v >= 0 else None) for k, v in
        n_observations_per_split.items()}
    self.target_lens = {'train': self.hparams.max_target_length, 'val':
        self.hparams.val_max_target_length, 'test': self.hparams.
        test_max_target_length}
    assert self.target_lens['train'] <= self.target_lens['val'
        ], f'target_lens: {self.target_lens}'
    assert self.target_lens['train'] <= self.target_lens['test'
        ], f'target_lens: {self.target_lens}'
    if self.hparams.freeze_embeds:
        freeze_embeds(self.model)
    if self.hparams.freeze_encoder:
        freeze_params(self.model.get_encoder())
        assert_all_frozen(self.model.get_encoder())
    self.num_workers = hparams.num_workers
    self.decoder_start_token_id = None
    if self.model.config.decoder_start_token_id is None and isinstance(self
        .tokenizer, MBartTokenizer):
        self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams
            .tgt_lang]
        self.model.config.decoder_start_token_id = self.decoder_start_token_id
    self.dataset_class = Seq2SeqDataset if hasattr(self.tokenizer,
        'prepare_seq2seq_batch') else LegacySeq2SeqDataset
    self.already_saved_batch = False
    self.eval_beams = (self.model.config.num_beams if self.hparams.
        eval_beams is None else self.hparams.eval_beams)
    if self.hparams.eval_max_gen_length is not None:
        self.eval_max_length = self.hparams.eval_max_gen_length
    else:
        self.eval_max_length = self.model.config.max_length
    self.val_metric = (self.default_val_metric if self.hparams.val_metric is
        None else self.hparams.val_metric)
    self.custom_pred_file_suffix = self.hparams.custom_pred_file_suffix
    self.do_sample = self.hparams.do_sample
    self.top_p = self.hparams.top_p
    self.top_k = self.hparams.top_k
    self.length_penalty = self.hparams.length_penalty
    self.temperature = self.hparams.temperature
    self.task_loss_ratio = self.hparams.task_loss_ratio
    self.extra_task = self.hparams.extra_task
    self.num_return_sequences = self.hparams.num_return_sequences
