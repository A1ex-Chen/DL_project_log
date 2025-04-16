def init_obj(self, name, module, *args, **kwargs):
    """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
    module_name = self[name]['type']
    module_args = dict(self[name]['args'])
    assert all([(k not in module_args) for k in kwargs]
        ), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)
