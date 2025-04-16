def generate_plane_features(self, index, c):
    c = c.permute(0, 2, 1)
    if index.max() < self.reso_plane ** 2:
        fea_plane = c.new_zeros(c.size(0), self.c_dim, self.reso_plane ** 2)
        fea_plane = scatter_mean(c, index, out=fea_plane)
    else:
        fea_plane = scatter_mean(c, index)
        if fea_plane.shape[-1] > self.reso_plane ** 2:
            fea_plane = fea_plane[:, :, :-1]
    fea_plane = fea_plane.reshape(c.size(0), self.c_dim, self.reso_plane,
        self.reso_plane)
    if self.unet is not None:
        fea_plane = self.unet(fea_plane)
    return fea_plane
