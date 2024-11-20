def __init__(self, in_dim=65, out_dim=64, c_dim=128, hidden_size=512, leaky
    =False, n_blocks=5, **kwargs):
    super().__init__()
    self.c_dim = c_dim
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.n_blocks = n_blocks
    self.fc_p = nn.Linear(in_dim, hidden_size)
    if c_dim != 0:
        self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in
            range(n_blocks)])
    self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(
        n_blocks)])
    self.fc_out = nn.Linear(hidden_size, self.out_dim)
    if not leaky:
        self.actvn = F.relu
    else:
        self.actvn = lambda x: F.leaky_relu(x, 0.2)
