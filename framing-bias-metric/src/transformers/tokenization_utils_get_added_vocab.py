def get_added_vocab(self) ->Dict[str, int]:
    """
        Returns the added tokens in the vocabulary as a dictionary of token to index.

        Returns:
            :obj:`Dict[str, int]`: The added tokens.
        """
    return self.added_tokens_encoder
