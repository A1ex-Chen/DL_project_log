def __init__(self, vocab, source_spm, target_spm, source_lang=None,
    target_lang=None, unk_token='<unk>', eos_token='</s>', pad_token=
    '<pad>', model_max_length=512, **kwargs):
    super().__init__(source_lang=source_lang, target_lang=target_lang,
        unk_token=unk_token, eos_token=eos_token, pad_token=pad_token,
        model_max_length=model_max_length, **kwargs)
    assert Path(source_spm).exists(), f'cannot find spm source {source_spm}'
    self.encoder = load_json(vocab)
    if self.unk_token not in self.encoder:
        raise KeyError('<unk> token must be in vocab')
    assert self.pad_token in self.encoder
    self.decoder = {v: k for k, v in self.encoder.items()}
    self.source_lang = source_lang
    self.target_lang = target_lang
    self.supported_language_codes: list = [k for k in self.encoder if k.
        startswith('>>') and k.endswith('<<')]
    self.spm_files = [source_spm, target_spm]
    self.spm_source = load_spm(source_spm)
    self.spm_target = load_spm(target_spm)
    self.current_spm = self.spm_source
    self._setup_normalizer()
