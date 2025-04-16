@property
def backend_tokenizer(self) ->TokenizerFast:
    """
        :obj:`tokenizers.implementations.BaseTokenizer`: The Rust tokenizer used as a backend.
        """
    return self._tokenizer
