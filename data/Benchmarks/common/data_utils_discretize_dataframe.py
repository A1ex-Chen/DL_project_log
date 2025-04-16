def discretize_dataframe(df, col, bins=2, cutoffs=None):
    """Discretize values of given column in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to process.
    col : int
        Index of column to bin.
    bins : int
        Number of bins for distributing column values.
    cutoffs : list
        List of bin limits.
        If None, the limits are computed as percentiles.
        (Default: None).

    Return
    ----------
    Returns the data frame with the values of the specified column binned, i.e. the values
    are replaced by the associated bin number.

    """
    y = df[col]
    thresholds = cutoffs
    if thresholds is None:
        percentiles = [(100 / bins * (i + 1)) for i in range(bins - 1)]
        thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    df[col] = classes
    return df
