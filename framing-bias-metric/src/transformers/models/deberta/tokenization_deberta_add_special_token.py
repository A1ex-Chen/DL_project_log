def add_special_token(self, token):
    """
        Adds a special token to the dictionary

        Args:
          token (:obj:`str`): Tthe new token/word to be added to the vocabulary.

        Returns:
          The id of new token in the vocabulary.

        """
    self.special_tokens.append(token)
    return self.add_symbol(token)
