def _convert_token_to_id(self, token):
    """ Converts a token (str/unicode) in an id using the vocab. """
    return self.vocab.get(token, self.vocab.get(self.unk_token))
