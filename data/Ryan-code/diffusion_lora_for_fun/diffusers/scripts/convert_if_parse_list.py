def parse_list(value):
    if isinstance(value, str):
        value = value.split(',')
        value = [int(v) for v in value]
    elif isinstance(value, list):
        pass
    else:
        raise ValueError(f"Can't parse list for type: {type(value)}")
    return value
