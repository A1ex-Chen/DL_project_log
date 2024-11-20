def init_ftn(self, name, module, *args, **kwargs):
    """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
    module_name = self[name]['type']
    module_args = dict(self[name]['args'])
    assert all([(k not in module_args) for k in kwargs]
        ), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return partial(getattr(module, module_name), *args, **module_args)
