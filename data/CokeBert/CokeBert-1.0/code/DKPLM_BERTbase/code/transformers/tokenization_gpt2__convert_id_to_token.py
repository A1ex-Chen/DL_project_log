def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (string/unicode) using the vocab."""
    return self.decoder.get(index)
