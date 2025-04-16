def _copy_layers(hf_layers, pt_layers):
    for i, hf_layer in enumerate(hf_layers):
        if i != 0:
            i += i
        pt_layer = pt_layers[i:i + 2]
        _copy_layer(hf_layer, pt_layer)
