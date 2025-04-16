def wrap(runner_cls):
    if name in cls.mapping['runner_name_mapping']:
        raise KeyError("Name '{}' already registered for {}.".format(name,
            cls.mapping['runner_name_mapping'][name]))
    cls.mapping['runner_name_mapping'][name] = runner_cls
    return runner_cls
