def _map(params, mask, label_fn):
    for k in params:
        if label_fn(k):
            mask[k] = 'token_embedding'
        elif isinstance(params[k], dict):
            mask[k] = {}
            _map(params[k], mask[k], label_fn)
        else:
            mask[k] = 'zero'
