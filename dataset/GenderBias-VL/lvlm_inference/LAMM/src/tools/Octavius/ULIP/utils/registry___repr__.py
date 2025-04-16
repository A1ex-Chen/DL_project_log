def __repr__(self):
    format_str = (self.__class__.__name__ +
        f'(name={self._name}, items={self._module_dict})')
    return format_str
