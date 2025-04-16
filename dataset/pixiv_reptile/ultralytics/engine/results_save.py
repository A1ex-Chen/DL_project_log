def save(self, filename=None, *args, **kwargs):
    """Save annotated inference results image to file."""
    if not filename:
        filename = f'results_{Path(self.path).name}'
    self.plot(*args, save=True, filename=filename, **kwargs)
    return filename
