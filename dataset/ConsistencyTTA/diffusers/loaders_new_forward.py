def new_forward(x):
    result = old_forward(x) + self.lora_scale * lora_layer(x)
    return result
