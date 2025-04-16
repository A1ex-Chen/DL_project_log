def __call__(self, module, inputs):
    """callable hook for forward pass"""
    module2use, name2use = Reparameterization.get_module_and_name(module,
        self.name)
    _w = getattr(module2use, name2use)
    if not self.evaluated or _w is None:
        setattr(module2use, name2use, self.compute_weight(module2use, name2use)
            )
        self.evaluated = True
