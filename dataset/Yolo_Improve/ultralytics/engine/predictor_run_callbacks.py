def run_callbacks(self, event: str):
    """Runs all registered callbacks for a specific event."""
    for callback in self.callbacks.get(event, []):
        callback(self)
