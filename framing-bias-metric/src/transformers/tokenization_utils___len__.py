def __len__(self):
    """
        Size of the full vocabulary with the added tokens.
        """
    return self.vocab_size + len(self.added_tokens_encoder)
