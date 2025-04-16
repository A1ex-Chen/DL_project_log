def create_custom_forward(module, return_dict=None):

    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)
    return custom_forward
