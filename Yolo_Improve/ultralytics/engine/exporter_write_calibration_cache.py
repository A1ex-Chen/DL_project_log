def write_calibration_cache(self, cache) ->None:
    """Write calibration cache to disk."""
    _ = self.cache.write_bytes(cache)
