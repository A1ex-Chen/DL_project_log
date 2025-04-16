def __init__(self, vocab_file, merges_file, tokenizer_file=None, unk_token=
    '<unk>', **kwargs):
    super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file,
        unk_token=unk_token, **kwargs)
