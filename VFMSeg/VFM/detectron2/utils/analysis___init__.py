def __init__(self, model, inputs):
    """
        Args:
            model (nn.Module):
            inputs (Any): inputs of the given model. Does not have to be tuple of tensors.
        """
    wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)
    super().__init__(wrapper, wrapper.flattened_inputs)
    self.set_op_handle(**{k: None for k in _IGNORED_OPS})
