def convert_ids_to_tokens(self, ids):
    """
        Convert list of ids to tokens

        Args:
          ids (:obj:`list<int>`): list of ids

        Returns:
          List of tokens
        """
    tokens = []
    for i in ids:
        tokens.append(self.ids_to_tokens[i])
    return tokens
