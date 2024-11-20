def __init__(self, vocab_file, merges_file, tokenizer_file=None, **kwargs):
    kwargs['cls_token'] = '<s>'
    kwargs['unk_token'] = '<unk>'
    kwargs['pad_token'] = '<pad>'
    kwargs['mask_token'] = '<mask>'
    kwargs['sep_token'] = '</s>'
    super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file,
        **kwargs)
