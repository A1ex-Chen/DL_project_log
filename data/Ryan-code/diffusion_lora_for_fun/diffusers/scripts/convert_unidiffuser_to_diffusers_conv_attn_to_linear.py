def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ['query.weight', 'key.weight', 'value.weight']
    for key in keys:
        if '.'.join(key.split('.')[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif 'proj_attn.weight' in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]
