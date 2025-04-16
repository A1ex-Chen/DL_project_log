def select_features_by_variation(data, variation_measure='var', threshold=
    None, portion=None, draw_histogram=False, bins=100, log=False):
    """
    This function evaluates the variations of individual features and returns the indices of features with large
    variations. Missing values are ignored in evaluating variation.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].
    variation_metric: string indicating the metric used for evaluating feature variation. 'var' indicates variance;
        'std' indicates standard deviation; 'mad' indicates median absolute deviation. Default is 'var'.
    threshold: float. Features with a variation larger than threshold will be selected. Default is None.
    portion: float in the range of [0, 1]. It is the portion of features to be selected based on variation.
        The number of selected features will be the smaller of int(portion * n_features) and the total number of
        features with non-missing variations. Default is None. threshold and portion can not take real values
        and be used simultaneously.
    draw_histogram: boolean, whether to draw a histogram of feature variations. Default is False.
    bins: positive integer, the number of bins in the histogram. Default is the smaller of 50 and the number of
        features with non-missing variations.
    log: boolean, indicating whether the histogram should be drawn on log scale.


    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected features. If both threshold and
        portion are None, indices will be an empty array.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)
    if variation_measure == 'std':
        v_all = np.nanstd(a=data, axis=0)
    elif variation_measure == 'mad':
        v_all = median_absolute_deviation(data=data, axis=0, ignore_nan=True)
    else:
        v_all = np.nanvar(a=data, axis=0)
    indices = np.where(np.invert(np.isnan(v_all)))[0]
    v = v_all[indices]
    if draw_histogram:
        if len(v) < 50:
            print(
                'There must be at least 50 features with variation measures to draw a histogram'
                )
        else:
            bins = int(min(bins, len(v)))
            _ = plt.hist(v, bins=bins, log=log)
            plt.show()
    if threshold is None and portion is None:
        return np.array([])
    elif threshold is not None and portion is not None:
        print(
            'threshold and portion can not be used simultaneously. Only one of them can take a real value'
            )
        sys.exit(1)
    if threshold is not None:
        indices = indices[np.where(v > threshold)[0]]
    else:
        n_f = int(min(portion * data.shape[1], len(v)))
        indices = indices[np.argsort(-v)[:n_f]]
    indices = np.sort(indices)
    return indices
