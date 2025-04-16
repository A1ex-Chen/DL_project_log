def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ['to_q.weight', 'to_k.weight', 'to_v.weight']
    proj_key = 'to_out.0.weight'
    for key in keys:
        if '.'.join(key.split('.')[-2:]) in attn_keys or '.'.join(key.split
            ('.')[-3:]) == proj_key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key].squeeze()
