def forward(self, xyz1, xyz2, points1, points2):
    """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
    xyz1 = xyz1.permute(0, 2, 1)
    xyz2 = xyz2.permute(0, 2, 1)
    points2 = points2.permute(0, 2, 1)
    B, N, C = xyz1.shape
    _, S, _ = xyz2.shape
    if S == 1:
        interpolated_points = points2.repeat(1, N, 1)
    else:
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :3], idx[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-08)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2, idx) * weight
            .view(B, N, 3, 1), dim=2)
    if points1 is not None:
        points1 = points1.permute(0, 2, 1)
        new_points = torch.cat([points1, interpolated_points], dim=-1)
    else:
        new_points = interpolated_points
    new_points = new_points.permute(0, 2, 1)
    for i, conv in enumerate(self.mlp_convs):
        bn = self.mlp_bns[i]
        new_points = F.relu(bn(conv(new_points)))
    return new_points
