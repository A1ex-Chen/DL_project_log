def has_next(self):
    """Whether the iterator has been exhausted."""
    return self.count < len(self)
