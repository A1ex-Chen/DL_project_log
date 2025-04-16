def eval_score(jacob, k=1e-05):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eig(corrs)
    return -(np.log(v + k) + 1.0 / (v + k))
