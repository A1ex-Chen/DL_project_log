def _get_return_layers(feature_info, out_map):
    module_names = feature_info.module_name()
    return_layers = {}
    for i, name in enumerate(module_names):
        return_layers[name] = out_map[i
            ] if out_map is not None else feature_info.out_indices[i]
    return return_layers
