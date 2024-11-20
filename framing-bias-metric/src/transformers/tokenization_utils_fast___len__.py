def __len__(self) ->int:
    """
        Size of the full vocabulary with the added tokens.
        """
    return self._tokenizer.get_vocab_size(with_added_tokens=True)
