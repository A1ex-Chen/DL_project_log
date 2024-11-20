def __init__(self, c1, k=1, s=1, r=16):
    super().__init__()
    c2 = max(r, c1 // r)
    self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
    self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
    self.fc1 = nn.Conv2d(c1, c2, k, s, bias=True)
    self.fc2 = nn.Conv2d(c2, c1, k, s, bias=True)
