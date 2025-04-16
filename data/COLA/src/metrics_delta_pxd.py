def delta_pxd(p_xd, p_dy, p_xy, **kwargs):
    return p_dy[0][0] - np.nanmean(p_xy[:, 0])
