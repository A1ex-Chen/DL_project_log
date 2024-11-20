def pool_local(self, index, c):
    bs, fea_dim = c.size(0), c.size(2)
    keys = index.keys()
    c_out = 0
    for key in keys:
        if key == 'grid':
            fea = self.scatter(c.permute(0, 2, 1), index[key])
        else:
            fea = self.scatter(c.permute(0, 2, 1), index[key])
        if self.scatter == scatter_max:
            fea = fea[0]
        fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
        c_out += fea
    return c_out.permute(0, 2, 1)
