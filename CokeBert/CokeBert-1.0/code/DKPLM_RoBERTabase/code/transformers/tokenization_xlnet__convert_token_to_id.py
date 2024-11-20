def _convert_token_to_id(self, token):
    """ Converts a token (str/unicode) in an id using the vocab. """
    return self.sp_model.PieceToId(token)
