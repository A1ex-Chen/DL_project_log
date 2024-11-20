def filter_float(value):
    if isinstance(value, str):
        return float(value.split()[0])
    return value
