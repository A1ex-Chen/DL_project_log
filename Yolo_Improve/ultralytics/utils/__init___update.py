def update(self, *args, **kwargs):
    """Updates a setting value in the current settings."""
    super().update(*args, **kwargs)
    self.save()
