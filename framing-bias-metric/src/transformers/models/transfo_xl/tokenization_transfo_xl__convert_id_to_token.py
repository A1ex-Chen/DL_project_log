def _convert_id_to_token(self, idx):
    """Converts an id in a token (BPE) using the vocab."""
    assert 0 <= idx < len(self), 'Index {} out of vocabulary range'.format(idx)
    return self.idx2sym[idx]
