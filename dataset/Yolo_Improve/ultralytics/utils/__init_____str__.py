def __str__(self):
    """Return a human-readable string representation of the object."""
    return '\n'.join(f'{k}={v}' for k, v in vars(self).items())
