def __init__(self, vocab_file, do_lower_case=True, remove_space=True,
    keep_accents=False, bos_token='[CLS]', eos_token='[SEP]', unk_token=
    '<unk>', sep_token='[SEP]', pad_token='<pad>', cls_token='[CLS]',
    mask_token='[MASK]', **kwargs):
    super().__init__(do_lower_case=do_lower_case, remove_space=remove_space,
        keep_accents=keep_accents, bos_token=bos_token, eos_token=eos_token,
        unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
        cls_token=cls_token, mask_token=mask_token, **kwargs)
    self.do_lower_case = do_lower_case
    self.remove_space = remove_space
    self.keep_accents = keep_accents
    self.vocab_file = vocab_file
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(vocab_file)
