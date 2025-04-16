def __init__(self, cfg):
    super(OfficialStaticNerf, self).__init__()
    D = cfg['model']['hidden_dim']
    pos_enc_levels = cfg['model']['pos_enc_levels']
    dir_enc_levels = cfg['model']['dir_enc_levels']
    pos_in_dims = (2 * pos_enc_levels + 1) * 3
    dir_in_dims = (2 * dir_enc_levels + 1) * 3
    self.white_bkgd = cfg['rendering']['white_background']
    self.dist_alpha = cfg['rendering']['dist_alpha']
    self.occ_activation = cfg['model']['occ_activation']
    self.layers0 = nn.Sequential(nn.Linear(pos_in_dims, D), nn.ReLU(), nn.
        Linear(D, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D
        ), nn.ReLU())
    self.layers1 = nn.Sequential(nn.Linear(D + pos_in_dims, D), nn.ReLU(),
        nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D), nn.ReLU(), nn.Linear(D,
        D), nn.ReLU())
    self.fc_density = nn.Linear(D, 1)
    self.fc_feature = nn.Linear(D, D)
    self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D // 2), nn.
        ReLU())
    self.fc_rgb = nn.Linear(D // 2, 3)
    self.fc_density.bias.data = torch.tensor([0.1]).float()
    self.sigmoid = nn.Sigmoid()
    if self.white_bkgd:
        self.fc_rgb.bias.data = torch.tensor([0.8, 0.8, 0.8]).float()
    else:
        self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()
