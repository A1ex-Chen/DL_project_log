def create_custom_forward(module):

    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward
