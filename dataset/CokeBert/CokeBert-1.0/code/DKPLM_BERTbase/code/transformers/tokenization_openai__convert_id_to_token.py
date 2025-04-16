def _convert_id_to_token(self, index):
    """Converts an id in a token (BPE) using the vocab."""
    return self.decoder.get(index, self.unk_token)
