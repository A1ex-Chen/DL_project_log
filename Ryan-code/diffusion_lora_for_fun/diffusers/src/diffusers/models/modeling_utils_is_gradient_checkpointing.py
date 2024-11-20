@property
def is_gradient_checkpointing(self) ->bool:
    """
        Whether gradient checkpointing is activated for this model or not.
        """
    return any(hasattr(m, 'gradient_checkpointing') and m.
        gradient_checkpointing for m in self.modules())
