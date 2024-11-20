def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.ids_to_tokens.get(index, self.unk_token)
