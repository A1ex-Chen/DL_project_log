def __init__(self):
    super(Model, self).__init__()
    self.a = Parameter(torch.cuda.FloatTensor(4096 * 4096).fill_(1.0))
    self.b = Parameter(torch.cuda.FloatTensor(4096 * 4096).fill_(2.0))
