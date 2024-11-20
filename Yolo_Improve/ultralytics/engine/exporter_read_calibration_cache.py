def read_calibration_cache(self) ->bytes:
    """Use existing cache instead of calibrating again, otherwise, implicitly return None."""
    if self.cache.exists() and self.cache.suffix == '.cache':
        return self.cache.read_bytes()
