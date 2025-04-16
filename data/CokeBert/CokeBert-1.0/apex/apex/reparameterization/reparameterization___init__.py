def __init__(self, name, dim, module, retain_forward=True):
    self.name = name
    self.dim = dim
    self.evaluated = False
    self.retain_forward = retain_forward
    self.reparameterization_names = []
    self.backward_hook_key = None
    self.module = module
