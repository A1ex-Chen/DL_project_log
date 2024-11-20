def discretize(y, bins=5, cutoffs=None, min_count=0, verbose=False,
    return_bins=False):
    thresholds = cutoffs
    if thresholds is None:
        if verbose:
            print('Creating {} balanced categories...'.format(bins))
        percentiles = [(100 / bins * (i + 1)) for i in range(bins - 1)]
        thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    good_bins = None
    if verbose:
        bc = np.bincount(classes)
        good_bins = len(bc)
        min_y = np.min(y)
        max_y = np.max(y)
        print('Category cutoffs:', ['{:.3g}'.format(t) for t in thresholds])
        print('Bin counts:')
        for i, count in enumerate(bc):
            lower = min_y if i == 0 else thresholds[i - 1]
            upper = max_y if i == len(bc) - 1 else thresholds[i]
            removed = ''
            if count < min_count:
                removed = ' .. removed (<{})'.format(min_count)
                good_bins -= 1
            print('  Class {}: {:7d} ({:.4f}) - between {:+.2f} and {:+.2f}{}'
                .format(i, count, count / len(y), lower, upper, removed))
    if return_bins:
        return classes, thresholds, good_bins
    else:
        return classes
