def set_weight_decay(model, skip_list=(), skip_keywords=(), echo=False):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'identity.weight' in name:
            has_decay.append(param)
            if echo:
                print(f'{name} USE weight decay')
        elif len(param.shape) == 1 or name.endswith('.bias'
            ) or name in skip_list or check_keywords_in_name(name,
            skip_keywords):
            no_decay.append(param)
            if echo:
                print(f'{name} has no weight decay')
        else:
            has_decay.append(param)
            if echo:
                print(f'{name} USE weight decay')
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.0}]
