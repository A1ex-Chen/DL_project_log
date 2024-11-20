def __init__(self, vocab_file, tokenizer_file=None, bos_token='<s>',
    eos_token='</s>', sep_token='</s>', cls_token='<s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>', additional_special_tokens=[
    '<s>NOTUSED', '</s>NOTUSED'], **kwargs):
    super().__init__(vocab_file, tokenizer_file=tokenizer_file, bos_token=
        bos_token, eos_token=eos_token, sep_token=sep_token, cls_token=
        cls_token, unk_token=unk_token, pad_token=pad_token, mask_token=
        mask_token, additional_special_tokens=additional_special_tokens, **
        kwargs)
    self.vocab_file = vocab_file
