def set(self, **kwargs):
    """
        Set multiple metadata with kwargs.
        """
    for k, v in kwargs.items():
        setattr(self, k, v)
    return self
