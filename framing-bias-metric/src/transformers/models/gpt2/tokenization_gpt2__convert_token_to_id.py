def _convert_token_to_id(self, token):
    """ Converts a token (str) in an id using the vocab. """
    return self.encoder.get(token, self.encoder.get(self.unk_token))
