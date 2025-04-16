def short_name(x):
    if len(x) > 13:
        return x[:11] + '..'
    return x
