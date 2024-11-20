def pack_weights(layers, prefix):
    layers_weights = layers.state_dict() if isinstance(layers, torch.nn.Module
        ) else layers
    layers_state_dict = {f'{prefix}.{module_name}': param for module_name,
        param in layers_weights.items()}
    return layers_state_dict
