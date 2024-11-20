def __init__(self):
    super(Odefunc, self).__init__()
    self.A = torch.nn.Parameter(torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]))
    self.unused_module = torch.nn.Linear(2, 5)
