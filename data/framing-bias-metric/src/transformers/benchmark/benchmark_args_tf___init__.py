def __init__(self, **kwargs):
    """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
    for deprecated_arg in self.deprecated_args:
        if deprecated_arg in kwargs:
            positive_arg = deprecated_arg[3:]
            kwargs[positive_arg] = not kwargs.pop(deprecated_arg)
            logger.warning(
                f'{deprecated_arg} is depreciated. Please use --no-{positive_arg} or {positive_arg}={kwargs[positive_arg]}'
                )
    self.tpu_name = kwargs.pop('tpu_name', self.tpu_name)
    self.device_idx = kwargs.pop('device_idx', self.device_idx)
    self.eager_mode = kwargs.pop('eager_mode', self.eager_mode)
    self.use_xla = kwargs.pop('use_xla', self.use_xla)
    super().__init__(**kwargs)
