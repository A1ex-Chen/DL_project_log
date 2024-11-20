def delta_bar(p_xd, p_dy, eps=0.01, ord=1, normalization=True, direct_match
    =False, use_cooccur=False, res_norm=False, pxa_all_norm=False):
    p_xd = deepcopy(p_xd)
    p_dy = deepcopy(p_dy)

    def adj(distribution, eps, ord, temp_filter=False):
        if temp_filter:
            distribution = distribution[distribution[:, 0, 0] >
                distribution[:, 0, 1]]
        Zn = distribution[:, :, 1] + distribution[:, :, 0]
        Zn[np.where(Zn == 0)] = 1
        before_prob = distribution[:, :, 0]
        if normalization:
            before_prob /= Zn
        p_x = before_prob[:, 0]
        if pxa_all_norm:
            p_xa = before_prob / np.sum(before_prob, axis=0)
            p_x = p_x / np.sum(before_prob[:, 0])
            p_xa[:, 0] = p_x
        else:
            p_xa = before_prob
        p_x[p_x == 0] = 1 / (before_prob.shape[0] if before_prob.shape[0] !=
            0 else 1)
        p_a_x = (p_xa.T / p_x.T).T
        matched_idx = return_match_idx(p_a_x, ord=ord, eps=eps)
        return matched_idx
    if direct_match:
        matched_idx = return_match_idx(p_xd[:, :, 0], ord=ord, eps=eps)
    else:
        matched_idx = adj(p_xd, eps=eps, ord=ord, temp_filter=False)
    if use_cooccur:
        sc = np.mean(p_dy[0, :2]) - (np.nanmean(p_dy[1:, :2][matched_idx]) if
            len(matched_idx) > 0 else 0)
    elif res_norm:

        def div_0arr(s):
            s[s == 0] = 1
            return s
        div_0 = lambda s: s if s != 0 else 1
        p_dy_proc = p_dy[1:, 0] / div_0arr(p_dy[1:, 0] + p_dy[1:, 1])
        p_dy_0 = p_dy[0, 0] / div_0(p_dy[0, 0] + p_dy[0, 1])
        sc = p_dy_0 - (np.nanmean(p_dy_proc[matched_idx]) if len(
            matched_idx) > 0 else 0)
    else:
        sc = p_dy[0, 0] - (np.nanmean(p_dy[1:, 0][matched_idx]) if len(
            matched_idx) > 0 else 0)
    return sc
