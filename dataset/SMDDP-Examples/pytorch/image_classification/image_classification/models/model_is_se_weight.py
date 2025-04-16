def is_se_weight(key, value):
    return key.endswith('squeeze.weight') or key.endswith('expand.weight')
