@errors.wrap_modelkit_exceptions
def __call__(self, item: ItemType, _force_compute: bool=False, **kwargs
    ) ->ReturnType:
    return self.predict(item, _force_compute=_force_compute, __internal=
        True, **kwargs)
