def __init__(self, vocab_file, tokenizer_file=None, eos_token='</s>',
    unk_token='<unk>', pad_token='<pad>', extra_ids=100,
    additional_special_tokens=None, **kwargs):
    if extra_ids > 0 and additional_special_tokens is None:
        additional_special_tokens = ['<extra_id_{}>'.format(i) for i in
            range(extra_ids)]
    elif extra_ids > 0 and additional_special_tokens is not None:
        extra_tokens = len(set(filter(lambda x: bool('extra_id_' in x),
            additional_special_tokens)))
        if extra_tokens != extra_ids:
            raise ValueError(
                f'Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are provided to T5Tokenizer. In this case the additional_special_tokens must include the extra_ids tokens'
                )
    super().__init__(vocab_file, tokenizer_file=tokenizer_file, eos_token=
        eos_token, unk_token=unk_token, pad_token=pad_token, extra_ids=
        extra_ids, additional_special_tokens=additional_special_tokens, **
        kwargs)
    self.vocab_file = vocab_file
    self._extra_ids = extra_ids
