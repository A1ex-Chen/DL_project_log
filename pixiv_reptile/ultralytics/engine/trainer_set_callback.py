def set_callback(self, event: str, callback):
    """Overrides the existing callbacks with the given callback."""
    self.callbacks[event] = [callback]
