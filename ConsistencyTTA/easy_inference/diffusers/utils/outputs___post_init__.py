def __post_init__(self):
    class_fields = fields(self)
    if not len(class_fields):
        raise ValueError(f'{self.__class__.__name__} has no fields.')
    first_field = getattr(self, class_fields[0].name)
    other_fields_are_none = all(getattr(self, field.name) is None for field in
        class_fields[1:])
    if other_fields_are_none and isinstance(first_field, dict):
        for key, value in first_field.items():
            self[key] = value
    else:
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v
