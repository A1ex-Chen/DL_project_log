def should_cache(x):
    if is_nested(x):
        for y in x:
            if not should_cache(y):
                return False
        return True
    return isinstance(x, torch.nn.parameter.Parameter) and type_string(x
        ) == 'FloatTensor'
