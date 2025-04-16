def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    K_XX = polynomial_kernel(features_1, features_1, degree=degree, gamma=
        gamma, coef0=coef0)
    K_YY = polynomial_kernel(features_2, features_2, degree=degree, gamma=
        gamma, coef0=coef0)
    K_XY = polynomial_kernel(features_1, features_2, degree=degree, gamma=
        gamma, coef0=coef0)
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)
    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()
    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    mmd2 -= 2 * K_XY_sum / (m * m)
    return mmd2
