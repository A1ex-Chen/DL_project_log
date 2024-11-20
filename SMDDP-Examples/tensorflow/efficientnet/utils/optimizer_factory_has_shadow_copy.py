@property
def has_shadow_copy(self):
    """Whether this optimizer has created shadow variables."""
    return self._model_weights is not None
