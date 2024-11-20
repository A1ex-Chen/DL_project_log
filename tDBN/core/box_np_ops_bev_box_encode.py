def bev_box_encode(boxes, anchors, encode_angle_to_vector=False, smooth_dim
    =False):
    """box encode for VoxelNet in lidar
    Args:
        boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
        anchors ([N, 7] Tensor): anchors
        encode_angle_to_vector: bool. increase aos performance,
            decrease other performance.
    """
    xa, ya, wa, la, ra = np.split(anchors, 5, axis=-1)
    xg, yg, wg, lg, rg = np.split(boxes, 5, axis=-1)
    diagonal = np.sqrt(la ** 2 + wa ** 2)
    xt = (xg - xa) / diagonal
    yt = (yg - ya) / diagonal
    if smooth_dim:
        lt = lg / la - 1
        wt = wg / wa - 1
    else:
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
    if encode_angle_to_vector:
        rgx = np.cos(rg)
        rgy = np.sin(rg)
        rax = np.cos(ra)
        ray = np.sin(ra)
        rtx = rgx - rax
        rty = rgy - ray
        return np.concatenate([xt, yt, wt, lt, rtx, rty], axis=-1)
    else:
        rt = rg - ra
        return np.concatenate([xt, yt, wt, lt, rt], axis=-1)
