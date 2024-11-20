def __init__(self, dim=3, c_dim=128, hidden_size=256, leaky=False, legacy=False
    ):
    super().__init__()
    self.dim = dim
    self.fc_p = nn.Conv1d(dim, hidden_size, 1)
    self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
    if not legacy:
        self.bn = CBatchNorm1d(c_dim, hidden_size)
    else:
        self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)
    self.fc_out = nn.Conv1d(hidden_size, 1, 1)
    if not leaky:
        self.actvn = F.relu
    else:
        self.actvn = lambda x: F.leaky_relu(x, 0.2)
