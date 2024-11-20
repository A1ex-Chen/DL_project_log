def register_to_config(self, **kwargs):
    if self.config_name is None:
        raise NotImplementedError(
            f'Make sure that {self.__class__} has defined a class name `config_name`'
            )
    kwargs.pop('kwargs', None)
    if not hasattr(self, '_internal_dict'):
        internal_dict = kwargs
    else:
        previous_dict = dict(self._internal_dict)
        internal_dict = {**self._internal_dict, **kwargs}
        logger.debug(f'Updating config from {previous_dict} to {internal_dict}'
            )
    self._internal_dict = FrozenDict(internal_dict)
