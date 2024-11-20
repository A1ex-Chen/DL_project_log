def is_saveable_module(name, value):
    if name not in expected_modules:
        return False
    if name in self._optional_components and value[0] is None:
        return False
    return True
