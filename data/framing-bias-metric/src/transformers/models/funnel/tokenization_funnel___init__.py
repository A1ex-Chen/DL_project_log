def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True,
    never_split=None, unk_token='<unk>', sep_token='<sep>', pad_token=
    '<pad>', cls_token='<cls>', mask_token='<mask>', bos_token='<s>',
    eos_token='</s>', tokenize_chinese_chars=True, strip_accents=None, **kwargs
    ):
    super().__init__(vocab_file, do_lower_case=do_lower_case,
        do_basic_tokenize=do_basic_tokenize, never_split=never_split,
        unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
        cls_token=cls_token, mask_token=mask_token, bos_token=bos_token,
        eos_token=eos_token, tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents, **kwargs)
