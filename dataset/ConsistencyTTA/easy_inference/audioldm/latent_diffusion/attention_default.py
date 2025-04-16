def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
