def check_backward_validity(inputs: Iterable[Any]) ->None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch
        .Tensor)):
        warnings.warn(
            'None of the inputs have requires_grad=True. Gradients will be None'
            )
