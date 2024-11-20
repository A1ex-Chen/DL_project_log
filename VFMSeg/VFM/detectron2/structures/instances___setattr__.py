def __setattr__(self, name: str, val: Any) ->None:
    if name.startswith('_'):
        super().__setattr__(name, val)
    else:
        self.set(name, val)
