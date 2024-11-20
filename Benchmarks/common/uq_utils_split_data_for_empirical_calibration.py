def split_data_for_empirical_calibration(Ytrue, Ypred, sigma, cal_split=0.8):
    """Extracts a portion of the arrays provided for the computation
    of the calibration and reserves the remainder portion
    for testing.

    Parameters
    ----------
    Ytrue : numpy array
        Array with true (observed) values
    Ypred : numpy array
        Array with predicted values.
    sigma : numpy array
        Array with standard deviations learned with deep learning
        model (or std value computed from prediction if homoscedastic
        inference).
    cal_split : float
         Split of data to use for estimating the calibration relationship.
         It is assumet that it will be a value in (0, 1).
         (Default: use 80% of predictions to generate empirical
         calibration).

    Return
    ----------
    index_perm_total : numpy array
        Random permutation of the array indices. The first 'num_cal'
        of the indices correspond to the samples that are used for
        calibration, while the remainder are the samples reserved
        for calibration testing.
    pSigma_cal : numpy array
        Part of the input sigma array to use for calibration.
    pSigma_test : numpy array
        Part of the input sigma array to reserve for testing.
    pPred_cal : numpy array
        Part of the input Ypred array to use for calibration.
    pPred_test : numpy array
        Part of the input Ypred array to reserve for testing.
    true_cal : numpy array
        Part of the input Ytrue array to use for calibration.
    true_test : numpy array
        Part of the input Ytrue array to reserve for testing.
    """
    num_pred_total = sigma.shape[0]
    num_cal = np.int(num_pred_total * cal_split)
    index_perm_total = np.random.permutation(range(num_pred_total))
    pSigma_perm_all = sigma[index_perm_total]
    pPred_perm_all = Ypred[index_perm_total]
    true_perm_all = Ytrue[index_perm_total]
    pSigma_cal = pSigma_perm_all[:num_cal]
    pSigma_test = pSigma_perm_all[num_cal:]
    pPred_cal = pPred_perm_all[:num_cal]
    pPred_test = pPred_perm_all[num_cal:]
    true_cal = true_perm_all[:num_cal]
    true_test = true_perm_all[num_cal:]
    print('Size of calibration set: ', true_cal.shape)
    print('Size of test set: ', true_test.shape)
    return (index_perm_total, pSigma_cal, pSigma_test, pPred_cal,
        pPred_test, true_cal, true_test)
