def __init__(self, vocab_file, bos_token='<s>', eos_token='</s>', unk_token
    ='<unk>', pad_token='<pad>', sep_token='<::::>', **kwargs):
    super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=
        unk_token, pad_token=pad_token, sep_token=sep_token, **kwargs)
    self.vocab_file = vocab_file
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(vocab_file)
