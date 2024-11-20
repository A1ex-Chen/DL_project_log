def __init__(self, vocab_file, bos_token='[SEP]', eos_token='[SEP]',
    sep_token='[SEP]', unk_token='[UNK]', pad_token='[PAD]', cls_token=
    '[CLS]', mask_token='[MASK]', **kwargs):
    super().__init__(bos_token=bos_token, eos_token=eos_token, sep_token=
        sep_token, unk_token=unk_token, pad_token=pad_token, cls_token=
        cls_token, mask_token=mask_token, **kwargs)
    try:
        import sentencepiece as spm
    except ImportError:
        logger.warning(
            'You need to install SentencePiece to use XLMRobertaTokenizer: https://github.com/google/sentencepiecepip install sentencepiece'
            )
        raise
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(str(vocab_file))
    self.vocab_file = vocab_file
    self.fairseq_tokens_to_ids = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2,
        '[UNK]': 3, '[MASK]': 4}
    for i in range(10):
        tok = '[unused{}]'.format(i)
        self.fairseq_tokens_to_ids[tok] = 5 + i
    self.fairseq_offset = 12
    self.fairseq_ids_to_tokens = {v: k for k, v in self.
        fairseq_tokens_to_ids.items()}
    for k in self.fairseq_tokens_to_ids.keys():
        self.unique_no_split_tokens.append(k)
