def _get_no_split_modules(self, device_map: str):
    """
        Get the modules of the model that should not be spit when using device_map. We iterate through the modules to
        get the underlying `_no_split_modules`.

        Args:
            device_map (`str`):
                The device map value. Options are ["auto", "balanced", "balanced_low_0", "sequential"]

        Returns:
            `List[str]`: List of modules that should not be split
        """
    _no_split_modules = set()
    modules_to_check = [self]
    while len(modules_to_check) > 0:
        module = modules_to_check.pop(-1)
        if module.__class__.__name__ not in _no_split_modules:
            if isinstance(module, ModelMixin):
                if module._no_split_modules is None:
                    raise ValueError(
                        f"{module.__class__.__name__} does not support `device_map='{device_map}'`. To implement support, the model class needs to implement the `_no_split_modules` attribute."
                        )
                else:
                    _no_split_modules = _no_split_modules | set(module.
                        _no_split_modules)
            modules_to_check += list(module.children())
    return list(_no_split_modules)
