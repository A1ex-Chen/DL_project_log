def discretize(df, col, bins=2, cutoffs=None):
    y = df[col]
    thresholds = cutoffs
    if thresholds is None:
        percentiles = [(100 / bins * (i + 1)) for i in range(bins - 1)]
        thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    df[col] = classes
    return df
