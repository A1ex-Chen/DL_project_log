def convert_attn_naming(name):
    for key, value in ATTN_MAP.items():
        if name.startswith(key) and not isinstance(value, list):
            return name.replace(key, value)
        elif name.startswith(key):
            return [name.replace(key, v) for v in value]
    raise ValueError(f'Attn error with {name}')
