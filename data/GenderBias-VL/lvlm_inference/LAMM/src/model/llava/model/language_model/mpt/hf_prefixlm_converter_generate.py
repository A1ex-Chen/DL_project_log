def generate(self: OPTForCausalLM, *args: tuple, **kwargs: Dict[str, Any]):
    """Wraps original generate to enable PrefixLM-style attention."""
    self.model.decoder.bidirectional_mask = 'g'
    try:
        output = self._original_generate(*args, **kwargs)
    except:
        self.model.decoder.bidirectional_mask = None
        raise
    self.model.decoder.bidirectional_mask = None
    return output
