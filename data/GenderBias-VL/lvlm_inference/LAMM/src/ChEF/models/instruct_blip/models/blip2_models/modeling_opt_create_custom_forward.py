def create_custom_forward(module):

    def custom_forward(*inputs):
        return module(*inputs, output_attentions, None)
    return custom_forward
