def backward_hook(self, module, grad_input, grad_output):
    """callable hook for backward pass"""
    module2use, name2use = Reparameterization.get_module_and_name(module,
        self.name)
    wn = getattr(module2use, name2use)
    self.evaluated = False
