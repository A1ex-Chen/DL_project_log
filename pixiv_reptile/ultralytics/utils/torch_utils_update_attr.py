def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
    """Updates attributes and saves stripped model with optimizer removed."""
    if self.enabled:
        copy_attr(self.ema, model, include, exclude)
