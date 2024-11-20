def __getattr__(self, name: str) ->Any:
    """The only reason we overwrite `getattr` here is to gracefully deprecate accessing
        config attributes directly. See https://github.com/huggingface/diffusers/pull/3129

        Tihs funtion is mostly copied from PyTorch's __getattr__ overwrite:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """
    is_in_config = '_internal_dict' in self.__dict__ and hasattr(self.
        __dict__['_internal_dict'], name)
    is_attribute = name in self.__dict__
    if is_in_config and not is_attribute:
        deprecation_message = (
            f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'scheduler.config.{name}'."
            )
        deprecate('direct config name access', '1.0.0', deprecation_message,
            standard_warn=False)
        return self._internal_dict[name]
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'")
