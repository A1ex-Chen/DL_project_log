def discretize(y, bins=5):
    percentiles = [(100 / bins * (i + 1)) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes
