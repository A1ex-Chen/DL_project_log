def __init__(self, num_classes, dual_head, backbone_3d, backbone_3d_kwargs):
    super(Net3DSeg, self).__init__()
    if backbone_3d == 'SCN':
        self.net_3d = UNetSCN(**backbone_3d_kwargs)
    else:
        raise NotImplementedError('3D backbone {} not supported'.format(
            backbone_3d))
    self.linear = nn.Linear(self.net_3d.out_channels, num_classes)
    self.dual_head = dual_head
    if dual_head:
        self.linear2 = nn.Linear(self.net_3d.out_channels, num_classes)
