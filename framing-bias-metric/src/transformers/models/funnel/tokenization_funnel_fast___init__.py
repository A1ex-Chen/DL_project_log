def __init__(self, vocab_file, tokenizer_file=None, do_lower_case=True,
    unk_token='<unk>', sep_token='<sep>', pad_token='<pad>', cls_token=
    '<cls>', mask_token='<mask>', bos_token='<s>', eos_token='</s>',
    clean_text=True, tokenize_chinese_chars=True, strip_accents=None,
    wordpieces_prefix='##', **kwargs):
    super().__init__(vocab_file, tokenizer_file=tokenizer_file,
        do_lower_case=do_lower_case, unk_token=unk_token, sep_token=
        sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=
        mask_token, bos_token=bos_token, eos_token=eos_token, clean_text=
        clean_text, tokenize_chinese_chars=tokenize_chinese_chars,
        strip_accents=strip_accents, wordpieces_prefix=wordpieces_prefix,
        **kwargs)
