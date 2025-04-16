def __init__(self, name, type_):
    assert isinstance(name, str), f'Field name must be str, got {name}'
    self.name = name
    self.type_ = type_
    self.annotation = f'{type_.__module__}.{type_.__name__}'
