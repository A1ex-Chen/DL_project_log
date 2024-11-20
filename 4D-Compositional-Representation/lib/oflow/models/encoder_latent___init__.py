def __init__(self, z_dim=128, c_dim=128, dim=51, hidden_dim=128, n_blocks=3,
    **kwargs):
    super().__init__()
    self.c_dim = c_dim
    self.dim = dim
    self.n_blocks = n_blocks
    self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
    self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for
        i in range(n_blocks)])
    if self.c_dim != 0:
        self.c_layers = nn.ModuleList([nn.Linear(c_dim, 2 * hidden_dim) for
            i in range(n_blocks)])
    self.actvn = nn.ReLU()
    self.pool = maxpool
    self.fc_mean = nn.Linear(hidden_dim, z_dim)
    self.fc_logstd = nn.Linear(hidden_dim, z_dim)
