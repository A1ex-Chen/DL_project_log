def __init__(self, device, dim=10):
    super(LinearODE, self).__init__()
    self.dim = dim
    U = torch.randn(dim, dim).to(device) * 0.1
    A = 2 * U - (U + U.transpose(0, 1))
    self.A = torch.nn.Parameter(A)
    self.initial_val = np.ones((dim, 1))
