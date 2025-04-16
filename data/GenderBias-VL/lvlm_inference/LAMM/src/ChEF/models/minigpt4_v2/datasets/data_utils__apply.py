def _apply(x):
    if torch.is_tensor(x):
        return f(x)
    elif isinstance(x, dict):
        return {key: _apply(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [_apply(x) for x in x]
    else:
        return x
