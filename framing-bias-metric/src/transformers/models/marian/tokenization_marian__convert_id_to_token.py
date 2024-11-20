def _convert_id_to_token(self, index: int) ->str:
    """Converts an index (integer) in a token (str) using the encoder."""
    return self.decoder.get(index, self.unk_token)
