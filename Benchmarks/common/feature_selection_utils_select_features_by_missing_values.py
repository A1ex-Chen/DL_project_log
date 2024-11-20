def select_features_by_missing_values(data, threshold=0.1):
    """
    This function returns the indices of the features whose missing rates are smaller than the threshold.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features]
    threshold: float in the range of [0, 1]. Features with a missing rate smaller than threshold will be selected.
        Default is 0.1

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected features
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)
    missing_rate = np.sum(np.isnan(data), axis=0) / data.shape[0]
    indices = np.where(missing_rate < threshold)[0]
    indices = np.sort(indices)
    return indices
