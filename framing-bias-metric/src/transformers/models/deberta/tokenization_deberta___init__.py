def __init__(self, vocab_file, do_lower_case=False, unk_token='[UNK]',
    sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token=
    '[MASK]', **kwargs):
    super().__init__(do_lower_case=do_lower_case, unk_token=unk_token,
        sep_token=sep_token, pad_token=pad_token, cls_token=cls_token,
        mask_token=mask_token, **kwargs)
    if not os.path.isfile(vocab_file):
        raise ValueError(
            "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained model use `tokenizer = XxxTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            .format(vocab_file))
    self.do_lower_case = do_lower_case
    self.gpt2_tokenizer = GPT2Tokenizer(vocab_file)
