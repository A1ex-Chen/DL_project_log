def tDBN_box_encode(boxes, anchors, encode_angle_to_vector=False,
    smooth_dim=False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    xa, ya, za, wa, la, ha, ra = np.split(anchors, 7, axis=-1)
    xg, yg, zg, wg, lg, hg, rg = np.split(boxes, 7, axis=-1)
    zg = zg + hg / 2
    za = za + ha / 2
    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    zt = (zg - za) / ha
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
        ht = hg / ha - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, zt, wt, lt, ht, rtx, rty], axis=-1)
    else:
        rt = rg - ra
        return np.concatenate([xt, yt, zt, wt, lt, ht, rt], axis=-1)
