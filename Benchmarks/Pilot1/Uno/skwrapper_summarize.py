def summarize(df, ycol='0', classify=False, bins=0, cutoffs=None, min_count=0):
    if ycol.isdigit():
        ycol = df.columns[int(ycol)]
    y = df.loc[:, ycol].values
    print('Target column: {}'.format(ycol))
    print('  count = {}, uniq = {}, mean = {:.3g}, std = {:.3g}'.format(len
        (y), len(np.unique(y)), np.mean(y), np.std(y)))
    print(
        '  min = {:.3g}, q1 = {:.3g}, median = {:.3g}, q3 = {:.3g}, max = {:.3g}'
        .format(np.min(y), np.percentile(y, 25), np.median(y), np.
        percentile(y, 75), np.max(y)))
    good_bins = None
    if classify:
        if cutoffs is not None or bins >= 2:
            _, _, good_bins = discretize(y, bins=bins, cutoffs=cutoffs,
                min_count=min_count, verbose=True)
        else:
            if df[ycol].dtype in [np.dtype('float64'), np.dtype('float32')]:
                warnings.warn(
                    'Warning: classification target is float; consider using --bins or --cutoffs'
                    )
            good_bins = len(np.unique(y))
    print()
    return good_bins
