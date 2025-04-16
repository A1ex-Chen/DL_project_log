def __init__(self, vocab_file, do_lower_case=False, remove_space=True,
    keep_accents=False, bos_token='<s>', eos_token='</s>', unk_token=
    '<unk>', sep_token='<sep>', pad_token='<pad>', cls_token='<cls>',
    mask_token='<mask>', additional_special_tokens=['<eop>', '<eod>'], **kwargs
    ):
    super(XLNetTokenizer, self).__init__(bos_token=bos_token, eos_token=
        eos_token, unk_token=unk_token, sep_token=sep_token, pad_token=
        pad_token, cls_token=cls_token, mask_token=mask_token,
        additional_special_tokens=additional_special_tokens, **kwargs)
    self.max_len_single_sentence = self.max_len - 2
    self.max_len_sentences_pair = self.max_len - 3
    try:
        import sentencepiece as spm
    except ImportError:
        logger.warning(
            'You need to install SentencePiece to use XLNetTokenizer: https://github.com/google/sentencepiecepip install sentencepiece'
            )
    self.do_lower_case = do_lower_case
    self.remove_space = remove_space
    self.keep_accents = keep_accents
    self.vocab_file = vocab_file
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(vocab_file)
