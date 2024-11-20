@property
def runtime(self):
    """Returns the model runtime."""
    return self.metadata.get('backend', self.metadata.get('platform'))
