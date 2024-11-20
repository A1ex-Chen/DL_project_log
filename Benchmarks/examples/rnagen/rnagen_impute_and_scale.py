def impute_and_scale(df, scaling='std', imputing='mean', dropna='all'):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to impute and scale
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """
    if dropna:
        df = df.dropna(axis=1, how=dropna)
    else:
        empty_cols = df.columns[df.notnull().sum() == 0]
        df[empty_cols] = 0
    if imputing is None or imputing.lower() == 'none':
        mat = df.values
    else:
        imputer = SimpleImputer(strategy=imputing)
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
