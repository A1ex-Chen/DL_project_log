def remap_key(key, sd):
    if key in sd or not peft_ge_070:
        return key
    if key.endswith('.weight'):
        key = key[:-7] + '.base_layer.weight'
    elif key.endswith('.bias'):
        key = key[:-5] + '.base_layer.bias'
    return key
