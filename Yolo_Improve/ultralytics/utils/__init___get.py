def get(self, key, default=None):
    """Return the value of the specified key if it exists; otherwise, return the default value."""
    return getattr(self, key, default)
