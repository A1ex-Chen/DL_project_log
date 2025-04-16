def __setattr__(self, name, value):
    if name in self.keys() and value is not None:
        super().__setitem__(name, value)
    super().__setattr__(name, value)
