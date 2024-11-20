def run_callbacks(self, event: str):
    """Run all existing callbacks associated with a particular event."""
    for callback in self.callbacks.get(event, []):
        callback(self)
