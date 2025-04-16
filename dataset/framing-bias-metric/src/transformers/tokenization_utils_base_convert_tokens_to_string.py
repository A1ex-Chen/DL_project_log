def convert_tokens_to_string(self, tokens: List[str]) ->str:
    """
        Converts a sequence of token ids in a single string. The most simple way to do it is ``" ".join(tokens)`` but
        we often want to remove sub-word tokenization artifacts at the same time

        Args:
            tokens (:obj:`List[str]`): The token to join in a string.
        Return: The joined tokens.
        """
    raise NotImplementedError
