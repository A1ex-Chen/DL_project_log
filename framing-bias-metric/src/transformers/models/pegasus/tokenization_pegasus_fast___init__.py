def __init__(self, vocab_file, tokenizer_file=None, pad_token='<pad>',
    eos_token='</s>', unk_token='<unk>', mask_token='<mask_2>',
    mask_token_sent='<mask_1>', additional_special_tokens=None, **kwargs):
    if additional_special_tokens is not None:
        assert isinstance(additional_special_tokens, list
            ), f'additional_special_tokens should be of type {type(list)}, but is {type(additional_special_tokens)}'
        additional_special_tokens_extended = ([mask_token_sent] +
            additional_special_tokens if mask_token_sent not in
            additional_special_tokens else additional_special_tokens)
        additional_special_tokens_extended += [f'<unk_{i}>' for i in range(
            len(additional_special_tokens_extended), self.offset - 1)]
        if len(set(additional_special_tokens_extended)) != len(
            additional_special_tokens_extended):
            raise ValueError(
                f'Please make sure that the provided additional_special_tokens do not contain an incorrectly shifted list of <unk_x> tokens. Found {additional_special_tokens_extended}.'
                )
        additional_special_tokens = additional_special_tokens_extended
    else:
        additional_special_tokens = [mask_token_sent]
        additional_special_tokens += [f'<unk_{i}>' for i in range(2, self.
            offset)]
    super().__init__(vocab_file, tokenizer_file=tokenizer_file, pad_token=
        pad_token, eos_token=eos_token, unk_token=unk_token, mask_token=
        mask_token, mask_token_sent=mask_token_sent,
        additional_special_tokens=additional_special_tokens, **kwargs)
    self.vocab_file = vocab_file
