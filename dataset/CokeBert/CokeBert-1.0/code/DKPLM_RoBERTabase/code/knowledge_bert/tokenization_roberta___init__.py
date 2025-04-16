def __init__(self, vocab_file, merges_file, errors='replace', bos_token=
    '<s>', eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token=
    '<unk>', pad_token='<pad>', mask_token='<mask>', **kwargs):
    super(RobertaTokenizer, self).__init__(vocab_file=vocab_file,
        merges_file=merges_file, errors=errors, bos_token=bos_token,
        eos_token=eos_token, unk_token=unk_token, sep_token=sep_token,
        cls_token=cls_token, pad_token=pad_token, mask_token=mask_token, **
        kwargs)
    self.max_len_single_sentence = self.max_len - 2
    self.max_len_sentences_pair = self.max_len - 4
