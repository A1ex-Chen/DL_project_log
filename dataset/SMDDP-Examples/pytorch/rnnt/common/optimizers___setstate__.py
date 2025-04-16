def __setstate__(self, state):
    super(Novograd, self).__setstate__(state)
    for group in self.param_groups:
        group.setdefault('amsgrad', False)
