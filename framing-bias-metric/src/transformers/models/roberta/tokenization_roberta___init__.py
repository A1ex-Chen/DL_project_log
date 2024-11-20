def __init__(self, vocab_file, merges_file, errors='replace', bos_token=
    '<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token=
    '<unk>', pad_token='<pad>', mask_token='<mask>', add_prefix_space=False,
    **kwargs):
    bos_token = AddedToken(bos_token, lstrip=False, rstrip=False
        ) if isinstance(bos_token, str) else bos_token
    eos_token = AddedToken(eos_token, lstrip=False, rstrip=False
        ) if isinstance(eos_token, str) else eos_token
    sep_token = AddedToken(sep_token, lstrip=False, rstrip=False
        ) if isinstance(sep_token, str) else sep_token
    cls_token = AddedToken(cls_token, lstrip=False, rstrip=False
        ) if isinstance(cls_token, str) else cls_token
    unk_token = AddedToken(unk_token, lstrip=False, rstrip=False
        ) if isinstance(unk_token, str) else unk_token
    pad_token = AddedToken(pad_token, lstrip=False, rstrip=False
        ) if isinstance(pad_token, str) else pad_token
    mask_token = AddedToken(mask_token, lstrip=True, rstrip=False
        ) if isinstance(mask_token, str) else mask_token
    super().__init__(vocab_file=vocab_file, merges_file=merges_file, errors
        =errors, bos_token=bos_token, eos_token=eos_token, unk_token=
        unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=
        pad_token, mask_token=mask_token, add_prefix_space=add_prefix_space,
        **kwargs)
