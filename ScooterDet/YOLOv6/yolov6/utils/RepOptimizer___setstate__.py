def __setstate__(self, state):
    super(SGD, self).__setstate__(state)
    for group in self.param_groups:
        group.setdefault('nesterov', False)
