def reset(self):
    """Resets the settings to default and saves them."""
    self.clear()
    self.update(self.defaults)
    self.save()
