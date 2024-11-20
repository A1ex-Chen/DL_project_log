def is_conditioned(self) ->bool:
    """Check whether all decoder layers are already conditioned."""
    return all(l.is_conditioned() for l in self._get_decoder_layers())
