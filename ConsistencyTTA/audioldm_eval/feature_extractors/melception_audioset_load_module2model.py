def load_module2model(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]
            new_state_dict[name] = v
    return new_state_dict
