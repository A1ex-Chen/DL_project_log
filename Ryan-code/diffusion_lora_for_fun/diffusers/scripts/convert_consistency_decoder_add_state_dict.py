def add_state_dict(prefix, mod):
    for k, v in mod.state_dict().items():
        unet_state_dict[f'{prefix}.{k}'] = v
