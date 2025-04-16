def coverage_80p(y_test, y_pred, std_pred, y_pred_1d=None, y_pred_9d=None):
    """Determine the fraction of the true data that falls
    into the 80p coverage of the model.
    For homoscedastic and heteroscedastic models the
    standard deviation prediction is used.
    For quantile model, if first and nineth deciles
    are available, these are used instead of computing
    a standard deviation based on Gaussian assumptions.

    Parameters
    ----------
    y_test : numpy array
        True (observed) values array.
    y_pred : numpy array
        Mean predictions made by the model.
    std_pred : numpy array
        Standard deviation predictions made by the model.
    y_pred_1d : numpy array
        First decile predictions made by qtl model.
    y_pred_9d : numpy array
        Nineth decile predictions made by qtl model.
    """
    if std_pred is None:
        topLim = y_pred_9d
        botLim = y_pred_1d
    else:
        topLim = y_pred + 1.28 * std_pred
        botLim = y_pred - 1.28 * std_pred
    count_gr = np.count_nonzero(np.clip(y_test - topLim, 0.0, None))
    count_ls = np.count_nonzero(np.clip(botLim - y_test, 0.0, None))
    count_out = count_gr + count_ls
    N_test = y_test.shape[0]
    frac_out = float(count_out) / float(N_test)
    return 1.0 - frac_out
