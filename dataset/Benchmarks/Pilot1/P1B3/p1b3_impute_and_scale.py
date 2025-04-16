def impute_and_scale(df, scaling='std'):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """
    df = df.dropna(axis=1, how='all')
    imputer = Imputer(strategy='mean')
    mat = imputer.fit_transform(df)
    if scaling is None or scaling.lower() == 'none':
        return pd.DataFrame(mat, columns=df.columns)
    if scaling == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    mat = scaler.fit_transform(mat)
    df = pd.DataFrame(mat, columns=df.columns)
    return df
