def map_from(module, state_dict, *args, **kwargs):
    all_keys = list(state_dict.keys())
    for key in all_keys:
        replace_key = remap_key(key, state_dict)
        new_key = key.replace(replace_key,
            f'layers.{module.rev_mapping[replace_key]}')
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
