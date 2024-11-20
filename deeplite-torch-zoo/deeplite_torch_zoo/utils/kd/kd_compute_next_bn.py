def compute_next_bn(layer_name, resnet):
    list_layer = extract_layers(resnet)
    assert layer_name in list_layer
    if layer_name == list_layer[-1]:
        return None
    next_bn = list_layer[list_layer.index(layer_name) + 1]
    if extract_layer(resnet, next_bn).__class__.__name__ == 'BatchNorm2d':
        return next_bn
    return None
