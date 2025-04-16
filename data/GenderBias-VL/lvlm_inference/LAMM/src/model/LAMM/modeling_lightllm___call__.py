def __call__(self, *args: Any, **kwds: Any) ->Any:
    return self.forward(*args, **kwds)
