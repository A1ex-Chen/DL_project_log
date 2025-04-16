def enable_gradient_checkpointing(self):
    """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
    if not self._supports_gradient_checkpointing:
        raise ValueError(
            f'{self.__class__.__name__} does not support gradient checkpointing.'
            )
    self.apply(partial(self._set_gradient_checkpointing, value=True))
