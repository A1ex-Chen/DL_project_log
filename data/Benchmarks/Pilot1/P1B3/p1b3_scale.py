def scale(df, scaling=None):
    """Scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to scale
    scaling : 'maxabs', 'minmax', 'std', or None, optional (default 'std')
        type of scaling to apply
    """
    if scaling is None or scaling.lower() == 'none':
        return df
    df = df.dropna(axis=1, how='any')
    if scaling == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    mat = df.as_matrix()
    mat = scaler.fit_transform(mat)
    df = pd.DataFrame(mat, columns=df.columns)
    return df
