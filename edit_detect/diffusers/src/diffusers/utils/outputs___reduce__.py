def __reduce__(self):
    if not is_dataclass(self):
        return super().__reduce__()
    callable, _args, *remaining = super().__reduce__()
    args = tuple(getattr(self, field.name) for field in fields(self))
    return callable, args, *remaining
