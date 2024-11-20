def __init__(self, *args, **kwargs):
    slow_tokenizer = kwargs.pop('__slow_tokenizer', None)
    fast_tokenizer_file = kwargs.pop('tokenizer_file', None)
    if fast_tokenizer_file is not None:
        fast_tokenizer = TokenizerFast.from_file(fast_tokenizer_file)
    elif slow_tokenizer is not None:
        fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
    elif self.slow_tokenizer_class is not None:
        slow_tokenizer = self.slow_tokenizer_class(*args, **kwargs)
        fast_tokenizer = convert_slow_tokenizer(slow_tokenizer)
    else:
        raise ValueError(
            "Couldn't instantiate the backend tokenizer from one of: (1) a `tokenizers` library serialization file, (2) a slow tokenizer instance to convert or (3) an equivalent slow tokenizer class to instantiate and convert. You need to have sentencepiece installed to convert a slow tokenizer to a fast one."
            )
    self._tokenizer = fast_tokenizer
    if slow_tokenizer is not None:
        kwargs.update(slow_tokenizer.init_kwargs)
    super().__init__(**kwargs)
