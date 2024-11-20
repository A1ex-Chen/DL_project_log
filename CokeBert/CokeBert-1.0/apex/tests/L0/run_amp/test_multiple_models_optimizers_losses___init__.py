def __init__(self, unique):
    super(MyModel, self).__init__()
    self.weight0 = Parameter(unique + torch.arange(2, device='cuda', dtype=
        torch.float32))
    self.weight1 = Parameter(1.0 + unique + torch.arange(2, device='cuda',
        dtype=torch.float16))
