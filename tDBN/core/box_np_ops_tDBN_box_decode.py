def tDBN_box_decode(box_encodings, anchors, encode_angle_to_vector=False,
    smooth_dim=False):
    """box decode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
    """
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    if encode_angle_to_vector:
        xt, yt, zt, wt, lt, ht, rtx, rty = np.split(box_encodings, 8, axis=-1)
    else:
        xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, 7, axis=-1)
    za = za + ha / 2
    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za
    if smooth_dim:
        lg = (lt + 1) * la
        wg = (wt + 1) * wa
        hg = (ht + 1) * ha
    else:
        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
    if encode_angle_to_vector:
        rax = np.cos(ra)
        ray = np.sin(ra)
        rgx = rtx + rax
        rgy = rty + ray
        rg = np.arctan2(rgy, rgx)
    else:
        rg = rt + ra
    zg = zg - hg / 2
    return np.concatenate([xg, yg, zg, wg, lg, hg, rg], axis=-1)
