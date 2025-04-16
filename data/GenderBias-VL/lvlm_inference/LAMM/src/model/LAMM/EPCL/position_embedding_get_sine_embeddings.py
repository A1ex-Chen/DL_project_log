def get_sine_embeddings(self, xyz, num_channels, input_range):
    orig_xyz = xyz
    xyz = orig_xyz.clone()
    ncoords = xyz.shape[1]
    if self.normalize:
        xyz = shift_scale_points(xyz, src_range=input_range)
    ndim = num_channels // xyz.shape[2]
    if ndim % 2 != 0:
        ndim -= 1
    rems = num_channels - ndim * xyz.shape[2]
    assert ndim % 2 == 0, f'Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}'
    final_embeds = []
    prev_dim = 0
    for d in range(xyz.shape[2]):
        cdim = ndim
        if rems > 0:
            cdim += 2
            rems -= 2
        if cdim != prev_dim:
            dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)
        raw_pos = xyz[:, :, d]
        if self.scale:
            raw_pos = raw_pos * self.scale
        pos = raw_pos[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
        final_embeds.append(pos)
        prev_dim = cdim
    final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
    return final_embeds
