def split_data(df, ycol='0', classify=False, cv=5, bins=0, cutoffs=None,
    groupcols=None, ignore_categoricals=False, verbose=True):
    if groupcols is not None:
        groups = make_group_from_columns(df, groupcols)
    cat_cols = df.select_dtypes(['object']).columns
    if ignore_categoricals:
        df[cat_cols] = 0
    else:
        df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category').
            cat.codes)
    if ycol.isdigit():
        ycol = df.columns[int(ycol)]
    y = df.loc[:, ycol].values
    x = df.drop(ycol, axis=1).values
    features = df.drop(ycol, axis=1).columns.tolist()
    if verbose:
        print('Target column: {}'.format(ycol))
        print('  count = {}, uniq = {}, mean = {:.3g}, std = {:.3g}'.format
            (len(y), len(np.unique(y)), np.mean(y), np.std(y)))
        print(
            '  min = {:.3g}, q1 = {:.3g}, median = {:.3g}, q3 = {:.3g}, max = {:.3g}'
            .format(np.min(y), np.percentile(y, 25), np.median(y), np.
            percentile(y, 75), np.max(y)))
    if not classify:
        y_even = discretize(y, bins=5, verbose=False)
    elif bins >= 2:
        y = discretize(y, bins=bins, min_count=cv, verbose=verbose)
    elif cutoffs:
        y = discretize(y, cutoffs=cutoffs, min_count=cv, verbose=verbose)
    elif df[ycol].dtype in [np.dtype('float64'), np.dtype('float32')]:
        warnings.warn(
            'Warning: classification target is float; consider using --bins or --cutoffs'
            )
        y = y.astype(int)
    if classify:
        mask = np.ones(len(y), dtype=bool)
        unique, counts = np.unique(y, return_counts=True)
        for v, c in zip(unique, counts):
            if c < cv:
                mask[y == v] = False
        x = x[mask]
        y = y[mask]
        removed = len(mask) - np.sum(mask)
        if removed and verbose:
            print('Removed {} rows in small classes: count < {}'.format(
                removed, cv))
    if groupcols is None:
        if classify:
            y_even = y
        skf = StratifiedKFold(n_splits=cv, shuffle=True)
        splits = skf.split(x, y_even)
    else:
        if classify:
            groups = groups[mask]
        gkf = GroupKFold(n_splits=cv)
        splits = gkf.split(x, y, groups)
    if verbose:
        print()
    return x, y, list(splits), features
