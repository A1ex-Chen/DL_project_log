def get_map_size(module, input, output):
    nonlocal map_size
    map_size = output.sample.shape[-2:]
