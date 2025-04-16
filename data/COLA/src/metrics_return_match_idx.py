def return_match_idx(mat, ord=None, eps=0.1, return_arr=False):
    d0 = mat[:, 0].T
    ds = mat[:, 1:].T
    Z = mat.shape[0]
    arr = np.linalg.norm(ds - d0, ord=ord, axis=1) / Z
    mat_idx = np.where((arr < eps) & (arr > 1e-08))[0]
    if return_arr:
        return mat_idx, arr
    return mat_idx
