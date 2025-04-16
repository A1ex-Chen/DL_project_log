def apply_hyperparameters(self, params):
    for attribute, param_values in params.items():
        for param_name, param_value in param_values.items():
            if attribute not in self.__dict__ and attribute != 'self':
                raise AttributeError(
                    f'attribute={attribute!r} not found in object')
            if attribute == 'self':
                logger.debug(
                    f'Setting {param_name} to {param_value} for {self}')
                setattr(self, param_name, param_value)
            else:
                attr_obj = getattr(self, attribute)
                if param_name in attr_obj.__dict__:
                    setattr(attr_obj, param_name, param_value)
                    logger.debug(
                        f'Setting {param_name} to {param_value} for {attribute}'
                        )
                else:
                    __dict__ = attr_obj.__dict__
                    raise TypeError(
                        f'Cannot set param_name={param_name!r} on attribute={attribute!r}, as it is immutable or not in {list(__dict__.keys())}'
                        )
