def convert_tokens_to_string(self, tokens):
    """ Converts a sequence of tokens (string) in a single string.
            The most simple way to do it is ' '.join(self.convert_ids_to_tokens(token_ids))
            but we often want to remove sub-word tokenization artifacts at the same time.
        """
    return ' '.join(self.convert_ids_to_tokens(tokens))
