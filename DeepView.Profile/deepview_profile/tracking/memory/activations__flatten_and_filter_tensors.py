def _flatten_and_filter_tensors(tensor_iterable):
    flattened = []
    for iterable_element in tensor_iterable:
        if isinstance(iterable_element, torch.Tensor
            ) and iterable_element.grad_fn is not None:
            flattened.append(iterable_element)
        elif isinstance(iterable_element, tuple) or isinstance(iterable_element
            , list):
            flattened.extend(_flatten_and_filter_tensors(iterable_element))
    return flattened
