def __init__(self, opt, module: nn.Module):
    super(BaseModel, self).__init__()
    self.opt = opt
    self.model = module
