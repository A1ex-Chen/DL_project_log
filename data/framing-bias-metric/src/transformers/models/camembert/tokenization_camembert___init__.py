def __init__(self, vocab_file, bos_token='<s>', eos_token='</s>', sep_token
    ='</s>', cls_token='<s>', unk_token='<unk>', pad_token='<pad>',
    mask_token='<mask>', additional_special_tokens=['<s>NOTUSED',
    '</s>NOTUSED'], **kwargs):
    super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=
        unk_token, sep_token=sep_token, cls_token=cls_token, pad_token=
        pad_token, mask_token=mask_token, additional_special_tokens=
        additional_special_tokens, **kwargs)
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(str(vocab_file))
    self.vocab_file = vocab_file
    self.fairseq_tokens_to_ids = {'<s>NOTUSED': 0, '<pad>': 1,
        '</s>NOTUSED': 2, '<unk>': 3}
    self.fairseq_offset = len(self.fairseq_tokens_to_ids)
    self.fairseq_tokens_to_ids['<mask>'] = len(self.sp_model) + len(self.
        fairseq_tokens_to_ids)
    self.fairseq_ids_to_tokens = {v: k for k, v in self.
        fairseq_tokens_to_ids.items()}
