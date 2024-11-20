def __init__(self, vocab_file, unk_token='<unk>', bos_token='<s>',
    eos_token='</s>', pad_token='</s>', sp_model_kwargs: Optional[Dict[str,
    Any]]=None, add_bos_token=True, add_eos_token=False,
    decode_with_prefix_space=False, clean_up_tokenization_spaces=False, **
    kwargs):
    self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
    self.vocab_file = vocab_file
    self.add_bos_token = add_bos_token
    self.add_eos_token = add_eos_token
    self.decode_with_prefix_space = decode_with_prefix_space
    self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
    self.sp_model.Load(vocab_file)
    self._no_prefix_space_tokens = None
    super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=
        unk_token, pad_token=pad_token, clean_up_tokenization_spaces=
        clean_up_tokenization_spaces, **kwargs)
    """ Initialization"""
