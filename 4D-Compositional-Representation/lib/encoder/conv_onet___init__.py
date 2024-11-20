def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max',
    unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
    plane_resolution=None, grid_resolution=None, plane_type='xz', padding=
    0.1, n_blocks=5, local_coord=False, pos_encoding='linear', unit_size=0.1):
    super().__init__()
    self.c_dim = c_dim
    self.blocks = nn.ModuleList([ResnetBlockFC(2 * hidden_dim, hidden_dim) for
        i in range(n_blocks)])
    self.fc_c = nn.Linear(hidden_dim, c_dim)
    self.actvn = nn.ReLU()
    self.hidden_dim = hidden_dim
    self.reso_plane = plane_resolution
    self.reso_grid = grid_resolution
    self.plane_type = plane_type
    self.padding = padding
    if unet:
        self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
    else:
        self.unet = None
    if unet3d:
        self.unet3d = UNet3D(**unet3d_kwargs)
    else:
        self.unet3d = None
    if scatter_type == 'max':
        self.scatter = scatter_max
    elif scatter_type == 'mean':
        self.scatter = scatter_mean
    else:
        raise ValueError('incorrect scatter type')
    if local_coord:
        self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
    else:
        self.map2local = None
    if pos_encoding == 'sin_cos':
        self.fc_pos = nn.Linear(60, 2 * hidden_dim)
    else:
        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
