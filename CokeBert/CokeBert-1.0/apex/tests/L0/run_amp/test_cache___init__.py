def __init__(self, dtype):
    super(PromoteModule, self).__init__()
    self.weight = torch.nn.Parameter(torch.arange(2 * 8, device='cuda',
        dtype=dtype).view(2, 8))
