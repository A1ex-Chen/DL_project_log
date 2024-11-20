def get_added_vocab(self) ->Dict[str, int]:
    """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            :obj:`Dict[str, int]`: The added tokens.
        """
    base_vocab = self._tokenizer.get_vocab(with_added_tokens=False)
    full_vocab = self._tokenizer.get_vocab(with_added_tokens=True)
    added_vocab = dict((tok, index) for tok, index in full_vocab.items() if
        tok not in base_vocab)
    return added_vocab
