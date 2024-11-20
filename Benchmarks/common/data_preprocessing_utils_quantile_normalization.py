def quantile_normalization(data):
    """
    This function does quantile normalization to input data. After normalization, the samples (rows) in output
    data follow the same distribution, which is the average distribution calculated based on all samples.
    This function allows missing values, and assume missing values occur at random.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].

    Returns:
    --------
    norm_data: numpy array or pandas data frame containing the data after quantile normalization.
    """
    colnames = None
    rownames = None
    if isinstance(data, pd.DataFrame):
        colnames = data.columns
        rownames = data.index
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)
    norm_data = data.copy()
    nan_mask = np.isnan(norm_data)
    if np.sum(nan_mask) > 0:
        n_samples, n_features = norm_data.shape
        for i in range(n_samples):
            idi_nan = np.where(np.isnan(norm_data[i, :]))[0]
            if len(idi_nan) > 0:
                idi = np.setdiff1d(range(n_features), idi_nan)
                norm_data[i, idi_nan] = np.random.choice(norm_data[i, idi],
                    size=len(idi_nan), replace=True)
    quantiles = np.mean(np.sort(norm_data, axis=1), axis=0)
    ranks = np.apply_along_axis(stats.rankdata, 1, norm_data)
    rank_indices = ranks.astype(int) - 1
    norm_data = quantiles[rank_indices]
    if np.sum(nan_mask) > 0:
        row_id, col_id = np.where(nan_mask)
        norm_data[row_id, col_id] = np.nan
    if colnames is not None and rownames is not None:
        norm_data = pd.DataFrame(norm_data, columns=colnames, index=rownames)
    return norm_data
