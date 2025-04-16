def disable_gradient_checkpointing(self) ->None:
    """
        Deactivates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
    if self._supports_gradient_checkpointing:
        self.apply(partial(self._set_gradient_checkpointing, value=False))
