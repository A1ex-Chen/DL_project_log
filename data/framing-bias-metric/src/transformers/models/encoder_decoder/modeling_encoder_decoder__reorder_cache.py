def _reorder_cache(self, past, beam_idx):
    return self.decoder._reorder_cache(past, beam_idx)
