@errors.wrap_modelkit_exceptions
def predict_batch(self, items: List[ItemType], _callback: Optional[Callable
    [[int, List[ItemType], List[ReturnType]], None]]=None, batch_size:
    Optional[int]=None, _force_compute: bool=False, **kwargs) ->List[ReturnType
    ]:
    batch_size = batch_size or (self.batch_size or len(items))
    return list(self.predict_gen(iter(items), _callback=_callback,
        batch_size=batch_size, _force_compute=_force_compute, __internal=
        True, **kwargs))
