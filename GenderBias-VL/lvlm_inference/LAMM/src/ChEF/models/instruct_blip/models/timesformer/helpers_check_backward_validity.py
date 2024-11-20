def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn(
            'None of the inputs have requires_grad=True. Gradients will be None'
            )
