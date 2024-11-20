def _cast_yaml_str(v):
    bool_dct = {'true': True, 'false': False}
    if not isinstance(v, str):
        return v
    elif v in bool_dct:
        return bool_dct[v]
    try:
        return int(v)
    except (TypeError, ValueError):
        return v
