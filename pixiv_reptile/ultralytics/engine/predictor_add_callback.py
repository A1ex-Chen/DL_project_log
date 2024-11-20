def add_callback(self, event: str, func):
    """Add callback."""
    self.callbacks[event].append(func)
