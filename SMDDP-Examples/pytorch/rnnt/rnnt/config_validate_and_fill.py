def validate_and_fill(klass, user_conf, ignore=[], optional=[]):
    conf = default_args(klass)
    for k, v in user_conf.items():
        assert k in conf or k in ignore, f'Unknown parameter {k} for {klass}'
        conf[k] = v
    conf = {k: v for k, v in conf.items() if k not in optional or v is not
        inspect.Parameter.empty}
    for k, v in conf.items():
        assert v is not inspect.Parameter.empty, f'Value for {k} not specified for {klass}'
    return conf
