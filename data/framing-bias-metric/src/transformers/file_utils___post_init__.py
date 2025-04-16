def __post_init__(self):
    class_fields = fields(self)
    assert len(class_fields), f'{self.__class__.__name__} has no fields.'
    assert all(field.default is None for field in class_fields[1:]
        ), f'{self.__class__.__name__} should not have more than one required field.'
    first_field = getattr(self, class_fields[0].name)
    other_fields_are_none = all(getattr(self, field.name) is None for field in
        class_fields[1:])
    if other_fields_are_none and not is_tensor(first_field):
        try:
            iterator = iter(first_field)
            first_field_iterator = True
        except TypeError:
            first_field_iterator = False
        if first_field_iterator:
            for element in iterator:
                if not isinstance(element, (list, tuple)) or not len(element
                    ) == 2 or not isinstance(element[0], str):
                    break
                setattr(self, element[0], element[1])
                if element[1] is not None:
                    self[element[0]] = element[1]
        elif first_field is not None:
            self[class_fields[0].name] = first_field
    else:
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v
