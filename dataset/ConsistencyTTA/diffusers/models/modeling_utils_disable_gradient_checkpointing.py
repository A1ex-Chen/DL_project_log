def disable_gradient_checkpointing(self):
    """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
    if self._supports_gradient_checkpointing:
        self.apply(partial(self._set_gradient_checkpointing, value=False))
