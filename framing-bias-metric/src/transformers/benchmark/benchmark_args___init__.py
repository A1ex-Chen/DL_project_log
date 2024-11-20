def __init__(self, **kwargs):
    """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
    for deprecated_arg in self.deprecated_args:
        if deprecated_arg in kwargs:
            positive_arg = deprecated_arg[3:]
            setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
            logger.warning(
                f'{deprecated_arg} is depreciated. Please use --no_{positive_arg} or {positive_arg}={kwargs[positive_arg]}'
                )
    self.torchscript = kwargs.pop('torchscript', self.torchscript)
    self.torch_xla_tpu_print_metrics = kwargs.pop('torch_xla_tpu_print_metrics'
        , self.torch_xla_tpu_print_metrics)
    self.fp16_opt_level = kwargs.pop('fp16_opt_level', self.fp16_opt_level)
    super().__init__(**kwargs)
