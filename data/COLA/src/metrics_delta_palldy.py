def delta_palldy(p_xd, p_dy, p_xy, **kwargs):
    matched_idx = return_match_idx(p_xd[:, :, 0], ord=1, eps=np.inf)
    return p_dy[0][0] - (np.nanmean(p_dy[1:, 0][matched_idx]) if len(
        matched_idx) > 0 else 0)
