def update_weight(x):
    if isinstance(x, dict):
        return {k: (v * w) for k, v in x.items()}
    else:
        return x * w
