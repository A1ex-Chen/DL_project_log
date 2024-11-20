def adj(distribution, eps, ord, temp_filter=False):
    if temp_filter:
        distribution = distribution[distribution[:, 0, 0] > distribution[:,
            0, 1]]
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
    p_x[p_x == 0] = 1 / (before_prob.shape[0] if before_prob.shape[0] != 0 else
        1)
    p_a_x = (p_xa.T / p_x.T).T
    matched_idx = return_match_idx(p_a_x, ord=ord, eps=eps)
    return matched_idx
