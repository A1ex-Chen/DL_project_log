def _use_cache(self, outputs, use_cache):
    """During generation, decide whether to pass the `past` variable to the next forward pass."""
    use_cache = getattr(self.config, 'use_cache', False)
    if len(outputs) <= 1 or use_cache is False:
        return False
    if hasattr(self.config, 'mem_len') and self.config.mem_len == 0:
        return False
    return True
