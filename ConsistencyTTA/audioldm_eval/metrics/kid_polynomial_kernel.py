def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma in [None, 'none', 'null', 'None']:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K
