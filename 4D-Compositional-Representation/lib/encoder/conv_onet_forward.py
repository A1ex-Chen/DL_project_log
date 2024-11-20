def forward(self, inputs):
    p = inputs['points']
    index = inputs['index']
    batch_size, T, D = p.size()
    if self.map2local:
        pp = self.map2local(p)
        net = self.fc_pos(pp)
    else:
        net = self.fc_pos(p)
    net = self.blocks[0](net)
    for block in self.blocks[1:]:
        pooled = self.pool_local(index, net)
        net = torch.cat([net, pooled], dim=2)
        net = block(net)
    c = self.fc_c(net)
    fea = {}
    if 'grid' in self.plane_type:
        fea['grid'] = self.generate_grid_features(index['grid'], c)
    if 'xz' in self.plane_type:
        fea['xz'] = self.generate_plane_features(index['xz'], c)
    if 'xy' in self.plane_type:
        fea['xy'] = self.generate_plane_features(index['xy'], c)
    if 'yz' in self.plane_type:
        fea['yz'] = self.generate_plane_features(index['yz'], c)
    return fea
