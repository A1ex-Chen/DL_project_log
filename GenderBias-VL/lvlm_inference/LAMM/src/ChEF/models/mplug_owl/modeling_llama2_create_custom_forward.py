def create_custom_forward(module):

    def custom_forward(*inputs):
        return module(*inputs, past_key_value, output_attentions)
    return custom_forward
