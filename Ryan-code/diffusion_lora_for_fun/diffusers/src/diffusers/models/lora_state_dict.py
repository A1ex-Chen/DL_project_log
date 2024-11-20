def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
    if self.lora_linear_layer is None:
        return self.regular_linear_layer.state_dict(*args, destination=
            destination, prefix=prefix, keep_vars=keep_vars)
    return super().state_dict(*args, destination=destination, prefix=prefix,
        keep_vars=keep_vars)
