def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
    super().__init__()
    self.c_dim = c_dim
    self.f_dim = f_dim
    self.norm_method = norm_method
    self.fc_gamma = nn.Linear(c_dim, f_dim)
    self.fc_beta = nn.Linear(c_dim, f_dim)
    if norm_method == 'batch_norm':
        self.bn = nn.BatchNorm1d(f_dim, affine=False)
    elif norm_method == 'instance_norm':
        self.bn = nn.InstanceNorm1d(f_dim, affine=False)
    elif norm_method == 'group_norm':
        self.bn = nn.GroupNorm1d(f_dim, affine=False)
    else:
        raise ValueError('Invalid normalization method!')
    self.reset_parameters()
