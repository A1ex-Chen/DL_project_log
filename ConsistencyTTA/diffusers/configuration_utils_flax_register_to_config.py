def flax_register_to_config(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f'`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does not inherit from `ConfigMixin`.'
                )
        init_kwargs = dict(kwargs.items())
        fields = dataclasses.fields(self)
        default_kwargs = {}
        for field in fields:
            if field.name in self._flax_internal_args:
                continue
            if type(field.default) == dataclasses._MISSING_TYPE:
                default_kwargs[field.name] = None
            else:
                default_kwargs[field.name] = getattr(self, field.name)
        new_kwargs = {**default_kwargs, **init_kwargs}
        if 'dtype' in new_kwargs:
            new_kwargs.pop('dtype')
        for i, arg in enumerate(args):
            name = fields[i].name
            new_kwargs[name] = arg
        getattr(self, 'register_to_config')(**new_kwargs)
        original_init(self, *args, **kwargs)
    cls.__init__ = init
    return cls
