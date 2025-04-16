def custom_forward(*inputs):
    if return_dict is not None:
        return module(*inputs, return_dict=return_dict)
    else:
        return module(*inputs)
