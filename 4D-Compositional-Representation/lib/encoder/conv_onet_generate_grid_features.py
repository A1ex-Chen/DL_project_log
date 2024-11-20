def generate_grid_features(self, index, c):
    c = c.permute(0, 2, 1)
    if index.max() < self.reso_grid ** 3:
        fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid ** 3)
        fea_grid = scatter_mean(c, index, out=fea_grid)
    else:
        fea_grid = scatter_mean(c, index)
        if fea_grid.shape[-1] > self.reso_grid ** 3:
            fea_grid = fea_grid[:, :, :-1]
    fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self
        .reso_grid, self.reso_grid)
    if self.unet3d is not None:
        fea_grid = self.unet3d(fea_grid)
    return fea_grid
