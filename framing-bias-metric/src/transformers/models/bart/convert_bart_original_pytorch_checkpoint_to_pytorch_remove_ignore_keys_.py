def remove_ignore_keys_(state_dict):
    ignore_keys = ['encoder.version', 'decoder.version',
        'model.encoder.version', 'model.decoder.version', '_float_tensor']
    for k in ignore_keys:
        state_dict.pop(k, None)
