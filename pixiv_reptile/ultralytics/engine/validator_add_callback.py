def add_callback(self, event: str, callback):
    """Appends the given callback."""
    self.callbacks[event].append(callback)
