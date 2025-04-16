def __init__(self, vocab_file, tokenizer_file=None, do_lower_case=True,
    unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token=
    '[CLS]', mask_token='[MASK]', tokenize_chinese_chars=True,
    strip_accents=None, **kwargs):
    super().__init__(vocab_file, tokenizer_file=tokenizer_file,
        do_lower_case=do_lower_case, unk_token=unk_token, sep_token=
        sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=
        mask_token, tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents, **kwargs)
    pre_tok_state = json.loads(self.backend_tokenizer.normalizer.__getstate__()
        )
    if pre_tok_state.get('do_lower_case', do_lower_case
        ) != do_lower_case or pre_tok_state.get('strip_accents', strip_accents
        ) != strip_accents:
        pre_tok_class = getattr(normalizers, pre_tok_state.pop('type'))
        pre_tok_state['do_lower_case'] = do_lower_case
        pre_tok_state['strip_accents'] = strip_accents
        self.backend_tokenizer.normalizer = pre_tok_class(**pre_tok_state)
    self.do_lower_case = do_lower_case
