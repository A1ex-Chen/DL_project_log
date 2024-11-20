def new_constructor(self, **kwargs):
    self.old_constructor(**kwargs)
    self.constructor_arguments = kwargs
