def _apply(self, fn):
    """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
    self = super()._apply(fn)
    m = self.model[-1]
    if isinstance(m, Detect):
        m.stride = fn(m.stride)
        m.anchors = fn(m.anchors)
        m.strides = fn(m.strides)
    return self
