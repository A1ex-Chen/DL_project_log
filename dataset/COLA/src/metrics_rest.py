"""
metrics.py
"""
import numpy as np
from copy import deepcopy











    if direct_match:
        matched_idx = return_match_idx(p_xd[:, :, 0], ord=ord, eps=eps)
    else:
        matched_idx = adj(p_xd, eps=eps, ord=ord, temp_filter=False)
    # print(matched_idx.reshape(-1).shape)

    if use_cooccur:
        sc = np.mean(p_dy[0, :2]) - (np.nanmean(p_dy[1:, :2][matched_idx]) if len(matched_idx) > 0 else 0)
    elif res_norm:

        div_0 = lambda s: s if s != 0 else 1
        p_dy_proc = p_dy[1:, 0] / div_0arr(p_dy[1:, 0] + p_dy[1:, 1])
        p_dy_0 = p_dy[0, 0] / div_0(p_dy[0, 0] + p_dy[0, 1])
        sc = p_dy_0 - (np.nanmean(p_dy_proc[matched_idx]) if len(matched_idx) > 0 else 0)
    else:
        sc = p_dy[0, 0] - (np.nanmean(p_dy[1:, 0][matched_idx]) if len(matched_idx) > 0 else 0)

    return sc


def delta_pxd(p_xd, p_dy, p_xy, **kwargs):
    return p_dy[0][0] - np.nanmean(p_xy[:, 0])


def delta_palldy(p_xd, p_dy, p_xy, **kwargs):
    matched_idx = return_match_idx(p_xd[:, :, 0], ord=1, eps=np.inf)
    return p_dy[0][0] - (np.nanmean(p_dy[1:, 0][matched_idx]) if len(matched_idx) > 0 else 0)


def delta_pdy(p_xd, p_dy, p_xy, **kwargs):
    return p_dy[0][0]