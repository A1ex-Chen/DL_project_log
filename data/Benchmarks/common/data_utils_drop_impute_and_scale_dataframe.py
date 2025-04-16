def drop_impute_and_scale_dataframe(df, scaling='std', imputing='mean',
    dropna='all'):
    """Impute missing values with mean and scale data included in pandas dataframe.

    Parameters
    ----------
    df : pandas dataframe
        dataframe to process
    scaling : string
        String describing type of scaling to apply.
        'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional
        (Default 'std')
    imputing : string
        String describing type of imputation to apply.
        'mean' replace missing values with mean value along the column,
        'median' replace missing values with median value along the column,
        'most_frequent' replace missing values with most frequent value along column
        (Default: 'mean').
    dropna : string
        String describing strategy for handling missing values.
        'all' if all values are NA, drop that column.
        'any' if any NA values are present, dropt that column.
        (Default: 'all').

    Return
    ----------
    Returns the data frame after handling missing values and scaling.

    """
    if dropna:
        df = df.dropna(axis=1, how=dropna)
    else:
        empty_cols = df.columns[df.notnull().sum() == 0]
        df[empty_cols] = 0
    if imputing is None or imputing.lower() == 'none':
        mat = df.values
    else:
        imputer = Imputer(strategy=imputing)
        mat = imputer.fit_transform(df.values)
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
