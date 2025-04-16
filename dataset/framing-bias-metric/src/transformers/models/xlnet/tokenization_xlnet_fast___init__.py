def __init__(self, vocab_file, tokenizer_file=None, do_lower_case=False,
    remove_space=True, keep_accents=False, bos_token='<s>', eos_token=
    '</s>', unk_token='<unk>', sep_token='<sep>', pad_token='<pad>',
    cls_token='<cls>', mask_token='<mask>', additional_special_tokens=[
    '<eop>', '<eod>'], **kwargs):
    super().__init__(vocab_file=vocab_file, tokenizer_file=tokenizer_file,
        do_lower_case=do_lower_case, remove_space=remove_space,
        keep_accents=keep_accents, bos_token=bos_token, eos_token=eos_token,
        unk_token=unk_token, sep_token=sep_token, pad_token=pad_token,
        cls_token=cls_token, mask_token=mask_token,
        additional_special_tokens=additional_special_tokens, **kwargs)
    self._pad_token_type_id = 3
    self.do_lower_case = do_lower_case
    self.remove_space = remove_space
    self.keep_accents = keep_accents
    self.vocab_file = vocab_file
