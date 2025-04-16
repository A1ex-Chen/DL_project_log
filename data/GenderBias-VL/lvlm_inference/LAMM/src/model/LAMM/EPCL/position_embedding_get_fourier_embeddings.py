def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
    if num_channels is None:
        num_channels = self.gauss_B.shape[1] * 2
    bsize, npoints = xyz.shape[0], xyz.shape[1]
    assert num_channels > 0 and num_channels % 2 == 0
    d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
    d_out = num_channels // 2
    assert d_out <= max_d_out
    assert d_in == xyz.shape[-1]
    orig_xyz = xyz
    xyz = orig_xyz.clone()
    ncoords = xyz.shape[1]
    if self.normalize:
        xyz = shift_scale_points(xyz, src_range=input_range)
    xyz = xyz * 2 * np.pi
    xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(bsize
        , npoints, d_out)
    final_embeds = [xyz_proj.sin(), xyz_proj.cos()]
    final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
    return final_embeds
