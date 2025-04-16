def create_custom_forward(module):

    def custom_forward(*inputs):
        return module(*inputs, output_attentions)
    return custom_forward
