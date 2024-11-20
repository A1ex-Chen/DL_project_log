def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True,
    never_split=None, unk_token='[UNK]', sep_token='[SEP]', x_sep_token=
    '[X_SEP]', pad_token='[PAD]', mask_token='[MASK]',
    tokenize_chinese_chars=True, strip_accents=None, **kwargs):
    super().__init__(do_lower_case=do_lower_case, do_basic_tokenize=
        do_basic_tokenize, never_split=never_split, unk_token=unk_token,
        sep_token=sep_token, x_sep_token=x_sep_token, pad_token=pad_token,
        mask_token=mask_token, tokenize_chinese_chars=
        tokenize_chinese_chars, strip_accents=strip_accents, **kwargs)
    self.unique_no_split_tokens.append(x_sep_token)
    if not os.path.isfile(vocab_file):
        raise ValueError(
            "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained model use `tokenizer = ProphetNetTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            .format(vocab_file))
    self.vocab = load_vocab(vocab_file)
    self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in
        self.vocab.items()])
    self.do_basic_tokenize = do_basic_tokenize
    if do_basic_tokenize:
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
            never_split=never_split, tokenize_chinese_chars=
            tokenize_chinese_chars, strip_accents=strip_accents)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
        unk_token=self.unk_token)
