def make_new_forward(old_forward, lora_layer):

    def new_forward(x):
        result = old_forward(x) + self.lora_scale * lora_layer(x)
        return result
    return new_forward
