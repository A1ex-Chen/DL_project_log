def __init__(self, vocab_file, merges_file, tokenizer_file=None, errors=
    'replace', bos_token='<s>', eos_token='</s>', sep_token='</s>',
    cls_token='<s>', unk_token='<unk>', pad_token='<pad>', mask_token=
    '<mask>', add_prefix_space=False, **kwargs):
    super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file,
        errors=errors, bos_token=bos_token, eos_token=eos_token, sep_token=
        sep_token, cls_token=cls_token, unk_token=unk_token, pad_token=
        pad_token, mask_token=mask_token, add_prefix_space=add_prefix_space,
        **kwargs)
