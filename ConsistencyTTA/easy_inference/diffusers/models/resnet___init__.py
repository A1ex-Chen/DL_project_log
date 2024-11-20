def __init__(self, in_dim, out_dim=None, dropout=0.0):
    super().__init__()
    out_dim = out_dim or in_dim
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.conv1 = nn.Sequential(nn.GroupNorm(32, in_dim), nn.SiLU(), nn.
        Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)))
    self.conv2 = nn.Sequential(nn.GroupNorm(32, out_dim), nn.SiLU(), nn.
        Dropout(dropout), nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1,
        0, 0)))
    self.conv3 = nn.Sequential(nn.GroupNorm(32, out_dim), nn.SiLU(), nn.
        Dropout(dropout), nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1,
        0, 0)))
    self.conv4 = nn.Sequential(nn.GroupNorm(32, out_dim), nn.SiLU(), nn.
        Dropout(dropout), nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1,
        0, 0)))
    nn.init.zeros_(self.conv4[-1].weight)
    nn.init.zeros_(self.conv4[-1].bias)
