def __init__(self, num):
    super().__init__()
    self.weight = torch.nn.Parameter(torch.tensor(num))
    self.is_run = False
    self.number = 0
    self.counter = 0
