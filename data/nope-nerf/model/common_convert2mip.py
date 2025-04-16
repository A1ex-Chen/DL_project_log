def convert2mip(pts):
    pts_norm = torch.linalg.norm(pts, ord=2, dim=-1)
    outside_mask = pts_norm >= 1.0
    mip_pts = pts.clone()
    mip_pts[outside_mask, :] = (2 - 1.0 / pts_norm[outside_mask, None]) * (pts
        [outside_mask, :] / pts_norm[outside_mask, None])
    return mip_pts
