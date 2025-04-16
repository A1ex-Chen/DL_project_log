def convert_tokens_to_ids(self, tokens):
    """
        Convert list of tokens to ids

        Args:
          tokens (:obj:`list<str>`): list of tokens

        Returns:
          List of ids
        """
    return [self.vocab[t] for t in tokens]
