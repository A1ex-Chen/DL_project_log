def __init__(self, num_classes, dual_head, backbone_2d, backbone_2d_kwargs,
    backbone_3d, backbone_3d_kwargs):
    super(Net2D3DFusionSeg, self).__init__()
    if backbone_2d == 'UNetResNet34':
        self.net_2d = UNetResNet34(**backbone_2d_kwargs)
        feat_channels_2d = 64
    else:
        raise NotImplementedError('2D backbone {} not supported'.format(
            backbone_2d))
    if backbone_3d == 'SCN':
        self.net_3d = UNetSCN(**backbone_3d_kwargs)
    else:
        raise NotImplementedError('3D backbone {} not supported'.format(
            backbone_3d))
    self.fuse = nn.Sequential(nn.Linear(feat_channels_2d + self.net_3d.
        out_channels, 64), nn.ReLU(inplace=True))
    self.linear = nn.Linear(64, num_classes)
    self.dual_head = dual_head
    if dual_head:
        self.linear_2d = nn.Linear(feat_channels_2d, num_classes)
        self.linear_3d = nn.Linear(self.net_3d.out_channels, num_classes)
