def __setitem__(self, key, value):
    super().__setitem__(key, value)
    super().__setattr__(key, value)
