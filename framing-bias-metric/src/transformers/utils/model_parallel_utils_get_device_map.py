def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i:i + n_blocks] for i in range(0, n_layers,
        n_blocks))
    return dict(zip(devices, layers_list))
