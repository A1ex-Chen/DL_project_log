def get_vocab(self) ->Dict[str, int]:
    """
        Returns the vocabulary as a dictionary of token to index.

        :obj:`tokenizer.get_vocab()[token]` is equivalent to :obj:`tokenizer.convert_tokens_to_ids(token)` when
        :obj:`token` is in the vocab.

        Returns:
            :obj:`Dict[str, int]`: The vocabulary.
        """
    raise NotImplementedError()
