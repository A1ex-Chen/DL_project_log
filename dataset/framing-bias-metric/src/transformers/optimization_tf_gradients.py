@property
def gradients(self):
    """The accumulated gradients on the current replica."""
    if not self._gradients:
        raise ValueError(
            'The accumulator should be called first to initialize the gradients'
            )
    return list(gradient.value() if gradient is not None else gradient for
        gradient in self._gradients)
