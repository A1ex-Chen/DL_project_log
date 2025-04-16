def __init__(self, vocab_file, eos_token='</s>', unk_token='<unk>',
    additional_special_tokens=[], **kwargs):
    super().__init__(eos_token=eos_token, unk_token=unk_token,
        additional_special_tokens=additional_special_tokens, **kwargs)
    self.vocab_file = vocab_file
    self.sp_model = spm.SentencePieceProcessor()
    self.sp_model.Load(vocab_file)
