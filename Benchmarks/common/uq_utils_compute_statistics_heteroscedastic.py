def compute_statistics_heteroscedastic(df_data, col_true=4, col_pred_start=
    6, col_std_pred_start=7):
    """Extracts ground truth, mean prediction, error, standard
    deviation of prediction and predicted (learned) standard
    deviation from inference data frame. The latter includes
    all the individual inference realizations.

    Parameters
    ----------
    df_data : pandas data frame
        Data frame generated by current heteroscedastic inference
        experiments. Indices are hard coded to agree with
        current version. (The inference file usually
        has the name: <model>.predicted_INFER_HET.tsv).
    col_true : integer
        Index of the column in the data frame where the true
        value is stored (Default: 4, index in current HET format).
    col_pred_start : integer
        Index of the column in the data frame where the first predicted
        value is stored. All the predicted values during inference
        are stored and are interspaced with standard deviation
        predictions (Default: 6 index, step 2, in current HET format).
    col_std_pred_start : integer
        Index of the column in the data frame where the first predicted
        standard deviation value is stored. All the predicted values
        during inference are stored and are interspaced with predictions
        (Default: 7 index, step 2, in current HET format).

    Return
    ----------
    Ytrue : numpy array
        Array with true (observed) values
    Ypred_mean : numpy array
        Array with predicted values (mean of predictions).
    yerror : numpy array
        Array with errors computed (observed - predicted).
    sigma : numpy array
        Array with standard deviations learned with deep learning
        model. For heteroscedastic inference this corresponds to the
        sqrt(exp(s^2)) with s^2 predicted value.
    Ypred_std : numpy array
        Array with standard deviations computed from regular
        (homoscedastic) inference.
    pred_name : string
        Name of data colum or quantity predicted (as extracted
        from the data frame using the col_true index).
    """
    Ytrue = df_data.iloc[:, col_true].values
    pred_name = df_data.columns[col_true]
    Ypred_mean_ = np.mean(df_data.iloc[:, col_pred_start::2], axis=1)
    Ypred_mean = Ypred_mean_.values
    Ypred_std_ = np.std(df_data.iloc[:, col_pred_start::2], axis=1)
    Ypred_std = Ypred_std_.values
    yerror = Ytrue - Ypred_mean
    s_ = df_data.iloc[:, col_std_pred_start::2]
    s_mean = np.mean(s_, axis=1)
    var = np.exp(s_mean.values)
    sigma = np.sqrt(var)
    MSE = mean_squared_error(Ytrue, Ypred_mean)
    print('MSE: ', MSE)
    MSE_STD = np.std((Ytrue - Ypred_mean) ** 2)
    print('MSE_STD: ', MSE_STD)
    MAE = mean_absolute_error(Ytrue, Ypred_mean)
    print('MAE: ', MAE)
    r2 = r2_score(Ytrue, Ypred_mean)
    print('R2: ', r2)
    pearson_cc, pval = pearsonr(Ytrue, Ypred_mean)
    print('Pearson CC: %f, p-value: %e' % (pearson_cc, pval))
    return Ytrue, Ypred_mean, yerror, sigma, Ypred_std, pred_name