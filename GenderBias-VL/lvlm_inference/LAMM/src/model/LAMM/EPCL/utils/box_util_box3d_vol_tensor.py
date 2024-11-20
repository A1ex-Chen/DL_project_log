def box3d_vol_tensor(corners):
    EPS = 1e-06
    reshape = False
    B, K = corners.shape[0], corners.shape[1]
    if len(corners.shape) == 4:
        reshape = True
        corners = corners.view(-1, 8, 3)
    a = torch.sqrt((corners[:, 0, :] - corners[:, 1, :]).pow(2).sum(dim=1).
        clamp(min=EPS))
    b = torch.sqrt((corners[:, 1, :] - corners[:, 2, :]).pow(2).sum(dim=1).
        clamp(min=EPS))
    c = torch.sqrt((corners[:, 0, :] - corners[:, 4, :]).pow(2).sum(dim=1).
        clamp(min=EPS))
    vols = a * b * c
    if reshape:
        vols = vols.view(B, K)
    return vols
