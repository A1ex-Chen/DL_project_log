def __init__(self, size):
    super(Foo, self).__init__()
    self.n = torch.nn.Parameter(torch.ones(size))
    self.m = torch.nn.Parameter(torch.ones(size))
