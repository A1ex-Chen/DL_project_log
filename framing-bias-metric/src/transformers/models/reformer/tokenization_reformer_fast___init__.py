def __init__(self, vocab_file, tokenizer_file=None, eos_token='</s>',
    unk_token='<unk>', additional_special_tokens=[], **kwargs):
    super().__init__(vocab_file, tokenizer_file=tokenizer_file, eos_token=
        eos_token, unk_token=unk_token, additional_special_tokens=
        additional_special_tokens, **kwargs)
    self.vocab_file = vocab_file
