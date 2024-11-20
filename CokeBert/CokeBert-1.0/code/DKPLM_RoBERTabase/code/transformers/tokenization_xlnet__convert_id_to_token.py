def _convert_id_to_token(self, index, return_unicode=True):
    """Converts an index (integer) in a token (string/unicode) using the vocab."""
    token = self.sp_model.IdToPiece(index)
    if six.PY2 and return_unicode and isinstance(token, str):
        token = token.decode('utf-8')
    return token
