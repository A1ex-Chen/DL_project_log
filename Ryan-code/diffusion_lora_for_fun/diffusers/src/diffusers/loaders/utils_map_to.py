def map_to(module, state_dict, *args, **kwargs):
    new_state_dict = {}
    for key, value in state_dict.items():
        num = int(key.split('.')[1])
        new_key = key.replace(f'layers.{num}', module.mapping[num])
        new_state_dict[new_key] = value
    return new_state_dict
