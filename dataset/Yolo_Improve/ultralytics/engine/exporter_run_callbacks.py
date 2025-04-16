def run_callbacks(self, event: str):
    """Execute all callbacks for a given event."""
    for callback in self.callbacks.get(event, []):
        callback(self)
