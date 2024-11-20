def __init__(self, in_channel, mlp):
    super(PointNetFeaturePropagation, self).__init__()
    self.mlp_convs = nn.ModuleList()
    self.mlp_bns = nn.ModuleList()
    last_channel = in_channel
    for out_channel in mlp:
        self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
        self.mlp_bns.append(nn.BatchNorm1d(out_channel))
        last_channel = out_channel
