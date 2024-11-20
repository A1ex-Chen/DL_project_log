def get_params(self, module):
    """gets params of reparameterization based on known attribute names"""
    return [getattr(module, n) for n in self.reparameterization_names]
