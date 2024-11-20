def __init__(self, c_dim=128, dim=51, hidden_dim=512, use_only_first_pcl=
    False, **kwargs):
    super().__init__()
    self.c_dim = c_dim
    self.use_only_first_pcl = use_only_first_pcl
    self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
    self.block_0 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
    self.block_1 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
    self.block_2 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
    self.block_3 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
    self.block_4 = ResnetBlockFC(2 * hidden_dim, hidden_dim)
    self.fc_c = nn.Linear(hidden_dim, c_dim)
    self.actvn = nn.ReLU()
    self.pool = maxpool
