def forward(self, p, ray_d=None, only_occupancy=False, return_logits=False,
    return_addocc=False, noise=False, it=100000, **kwargs):
    """
        :param pos_enc: (H, W, N_sample, pos_in_dims)
        :param dir_enc: (H, W, N_sample, dir_in_dims)
        :return: rgb_density (H, W, N_sample, 4)
        """
    x, density = self.infer_occ(p)
    if self.occ_activation == 'softplus':
        density = F.softplus(density)
    else:
        density = density.relu()
    if not self.dist_alpha:
        density = 1 - torch.exp(-1.0 * density)
    if only_occupancy:
        return density
    elif ray_d is not None:
        dir_enc = encode_position(ray_d, levels=4, inc_input=True)
        feat = self.fc_feature(x)
        x = torch.cat([feat, dir_enc], dim=-1)
        x = self.rgb_layers(x)
        rgb = self.fc_rgb(x)
        rgb = self.sigmoid(rgb)
        if return_addocc:
            return rgb, density
        else:
            return rgb
