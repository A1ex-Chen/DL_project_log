@property
def vocab_size(self) ->int:
    """
        :obj:`int`: Size of the base vocabulary (without the added tokens).
        """
    return self._tokenizer.get_vocab_size(with_added_tokens=False)
