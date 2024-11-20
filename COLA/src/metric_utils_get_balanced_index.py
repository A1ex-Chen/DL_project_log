def get_balanced_index(p_xd, p_dy, p_xy, eps=0.01, ord=1, normalization=
    True, direct_match=False, use_cooccur=False, temp_filter=False,
    pxa_all_norm=False, **kwargs):

    def adj(tprobs, eps, ord, temp_filter=False):
        if temp_filter:
            tprobs = tprobs[tprobs[:, 0, 0] > tprobs[:, 0, 1]]
        Zn = tprobs[:, :, 1] + tprobs[:, :, 0] + tprobs[:, :, 2] + tprobs[:,
            :, 3]
        Zn[np.where(Zn == 0)] = 1
        dat = tprobs[:, :, 0]
        if normalization:
            dat /= Zn
        p_x = dat[:, 0] / np.sum(dat[:, 0])
        if pxa_all_norm:
            p_xa = dat / np.sum(dat, axis=0)
        else:
            p_xa = dat / np.sum(dat[:, 0])
        p_x[p_x == 0] = 1 / (dat.shape[0] if dat.shape[0] != 0 else 1)
        p_a_x = (p_xa.T / p_x.T).T
        matched_idx_arr = return_match_idx(p_a_x, ord=ord, eps=eps,
            return_arr=True)
        return matched_idx_arr
    if direct_match:
        matched_idx_arr = return_match_idx(p_xd[:, :, 0], ord=ord, eps=eps,
            return_arr=True)
    else:
        matched_idx_arr = adj(p_xd, eps=eps, ord=ord, temp_filter=False, **
            kwargs)
    return matched_idx_arr
