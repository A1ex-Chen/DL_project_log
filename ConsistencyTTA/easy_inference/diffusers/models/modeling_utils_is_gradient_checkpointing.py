@property
def is_gradient_checkpointing(self) ->bool:
    """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
    return any(hasattr(m, 'gradient_checkpointing') and m.
        gradient_checkpointing for m in self.modules())
