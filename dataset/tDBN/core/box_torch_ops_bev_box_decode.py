def bev_box_decode(box_encodings, anchors, encode_angle_to_vector=False,
    smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, wa, la, ra = torch.split(anchors, 1, dim=-1)
    if encode_angle_to_vector:
        xt, yt, wt, lt, rtx, rty = torch.split(box_encodings, 1, dim=-1)
    else:
        xt, yt, wt, lt, rt = torch.split(box_encodings, 1, dim=-1)
    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
    else:
        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
    if encode_angle_to_vector:
        rax = torch.cos(ra)
        ray = torch.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = torch.atan2(rgy, rgx)
    else:
        rg = rt + ra
    return torch.cat([xg, yg, wg, lg, rg], dim=-1)
