def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = strtobool(value)
        except ValueError:
            raise ValueError('If set, {} must be yes or no.'.format(key))
    return _value
