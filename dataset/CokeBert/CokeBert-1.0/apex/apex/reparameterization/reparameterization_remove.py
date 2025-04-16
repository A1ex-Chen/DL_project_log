def remove(self, module):
    """removes reparameterization and backward hook (does not remove forward hook)"""
    module2use, name2use = Reparameterization.get_module_and_name(module,
        self.name)
    for p in self.get_params(module2use):
        p.requires_grad = False
    weight = self.compute_weight(module2use, name2use)
    delattr(module2use, name2use)
    for n in self.reparameterization_names:
        del module2use._parameters[n]
    module2use.register_parameter(name2use, Parameter(weight.data))
    del module._backward_hooks[self.backward_hook_key]
