def _convert_token_to_id(self, token: str) ->int:
    """ Converts a token (str) to an id using the vocab. """
    if token in self.decoder:
        return self.decoder[token]
    elif token in self.added_tokens_decoder:
        return self.added_tokens_decoder[token]
    sp_id = self.sp_model.piece_to_id(token)
    return sp_id + self.offset
