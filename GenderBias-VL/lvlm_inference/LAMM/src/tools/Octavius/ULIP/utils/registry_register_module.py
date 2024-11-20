def register_module(self, name=None, force=False, module=None):
    """Register a module.
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module(name='mnet')
            >>> class MobileNet:
            >>>     pass
            >>> backbones = Registry('backbone')
            >>> class ResNet:
            >>>     pass
            >>> backbones.register_module(ResNet)
        Args:
            name (str | None): The module name to be registered. If not
                specified, the class name will be used.
            force (bool, optional): Whether to override an existing class with
                the same name. Default: False.
            module (type): Module class to be registered.
        """
    if not isinstance(force, bool):
        raise TypeError(f'force must be a boolean, but got {type(force)}')
    if isinstance(name, type):
        return self.deprecated_register_module(name, force=force)
    if not (name is None or isinstance(name, str) or misc.is_seq_of(name, str)
        ):
        raise TypeError(
            f'name must be either of None, an instance of str or a sequence  of str, but got {type(name)}'
            )
    if module is not None:
        self._register_module(module_class=module, module_name=name, force=
            force)
        return module

    def _register(cls):
        self._register_module(module_class=cls, module_name=name, force=force)
        return cls
    return _register
