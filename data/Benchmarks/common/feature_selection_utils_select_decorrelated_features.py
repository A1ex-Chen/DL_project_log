def select_decorrelated_features(data, method='pearson', threshold=None,
    random_seed=None):
    """
    This function selects features whose mutual absolute correlation coefficients are smaller than a threshold.
    It allows missing values in data. The correlation coefficient of two features are calculated based on
    the observations that are not missing in both features. Features with only one or no value present and
    features with a zero standard deviation are not considered for selection.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].
    method: string indicating the method used for calculating correlation coefficient. 'pearson' indicates Pearson
        correlation coefficient; 'kendall' indicates Kendall Tau correlation coefficient; 'spearman' indicates
        Spearman rank correlation coefficient. Default is 'pearson'.
    threshold: float. If two features have an absolute correlation coefficient higher than threshold,
        one of the features is removed. If threshold is None, a feature is removed only when the two features
        are exactly identical. Default is None.
    random_seed: positive integer, seed of random generator for ordering the features. If it is None, features
        are not re-ordered before feature selection and thus the first feature is always selected. Default is None.

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected features.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif not isinstance(data, pd.DataFrame):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)
    present = np.where(np.sum(np.invert(pd.isna(data)), axis=0) > 1)[0]
    present = present[np.where(np.nanstd(data.iloc[:, present].values, axis
        =0) > 0)[0]]
    data = data.iloc[:, present]
    num_f = data.shape[1]
    if random_seed is not None:
        np.random.seed(random_seed)
        random_order = np.random.permutation(num_f)
        data = data.iloc[:, random_order]
    if threshold is not None:
        if np.sum(pd.isna(data).values) == 0 and method == 'pearson':
            cor = np.corrcoef(data.values, rowvar=False)
        else:
            cor = data.corr(method=method).values
    else:
        data = data.values
    rm = np.full(num_f, False)
    index = 0
    while index < num_f - 1:
        if rm[index]:
            index += 1
            continue
        idi = np.array(range(index + 1, num_f))
        idi = idi[np.where(rm[idi] == False)[0]]
        if len(idi) > 0:
            if threshold is None:
                idi = idi[np.where(np.sum(np.isnan(data[:, idi]) ^ np.isnan
                    (data[:, index][:, np.newaxis]), axis=0) == 0)[0]]
                if len(idi) > 0:
                    idi = idi[np.where(np.nansum(abs(data[:, idi] - data[:,
                        index][:, np.newaxis]), axis=0) == 0)[0]]
            else:
                idi = idi[np.where(abs(cor[index, idi]) >= threshold)[0]]
            if len(idi) > 0:
                rm[idi] = True
        index += 1
    indices = np.where(rm == False)[0]
    if random_seed is not None:
        indices = random_order[indices]
    indices = np.sort(present[indices])
    return indices
