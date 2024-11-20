def discretize_array(y, bins=5):
    """Discretize values of given array.

    Parameters
    ----------
    y : numpy array
        array to discretize.
    bins : int
        Number of bins for distributing column values.

    Return
    ----------
    Returns an array with the bin number associated to the values in the
    original array.

    """
    percentiles = [(100 / bins * (i + 1)) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes
