@errors.wrap_modelkit_exceptions
def predict(self, item: ItemType, _force_compute: bool=False, **kwargs
    ) ->ReturnType:
    return next(self.predict_gen(iter((item,)), _force_compute=
        _force_compute, __internal=True, **kwargs))
