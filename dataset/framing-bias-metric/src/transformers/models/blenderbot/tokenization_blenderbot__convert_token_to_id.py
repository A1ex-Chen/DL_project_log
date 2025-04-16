def _convert_token_to_id(self, token: str) ->int:
    """ Converts a token to an id using the vocab. """
    token = token.lower()
    return self.encoder.get(token, self.encoder.get(self.unk_token))
