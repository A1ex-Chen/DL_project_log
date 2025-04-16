def __init__(self, **kwargs):
    kwargs['cls_token'] = '<s>'
    kwargs['unk_token'] = '<unk>'
    kwargs['pad_token'] = '<pad>'
    kwargs['mask_token'] = '<mask>'
    kwargs['sep_token'] = '</s>'
    kwargs['do_lowercase_and_remove_accent'] = False
    kwargs['additional_special_tokens'] = []
    super().__init__(**kwargs)
    self.bert_pre_tokenizer = BasicTokenizer(do_lower_case=False,
        never_split=self.all_special_tokens, tokenize_chinese_chars=False,
        strip_accents=False)
