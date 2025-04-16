def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = strtobool(value)
        except ValueError:
            raise ValueError(f'If set, {key} must be yes or no.')
    return _value
