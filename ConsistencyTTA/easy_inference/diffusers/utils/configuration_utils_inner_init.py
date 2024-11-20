@functools.wraps(init)
def inner_init(self, *args, **kwargs):
    init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}
    config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith('_')}
    if not isinstance(self, ConfigMixin):
        raise RuntimeError(
            f'`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does not inherit from `ConfigMixin`.'
            )
    ignore = getattr(self, 'ignore_for_config', [])
    new_kwargs = {}
    signature = inspect.signature(init)
    parameters = {name: p.default for i, (name, p) in enumerate(signature.
        parameters.items()) if i > 0 and name not in ignore}
    for arg, name in zip(args, parameters.keys()):
        new_kwargs[name] = arg
    new_kwargs.update({k: init_kwargs.get(k, default) for k, default in
        parameters.items() if k not in ignore and k not in new_kwargs})
    new_kwargs = {**config_init_kwargs, **new_kwargs}
    getattr(self, 'register_to_config')(**new_kwargs)
    init(self, *args, **init_kwargs)
