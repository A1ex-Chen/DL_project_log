def run_callbacks(self, event: str):
    """Runs all callbacks associated with a specified event."""
    for callback in self.callbacks.get(event, []):
        callback(self)
