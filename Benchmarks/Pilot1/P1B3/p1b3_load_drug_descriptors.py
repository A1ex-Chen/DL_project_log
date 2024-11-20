def load_drug_descriptors(path, dtype, ncols=None, scaling='std'):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    path: string
        path to 'descriptors.2D-NSC.5dose.filtered.txt'
    dtype: numpy type
        precision (data type) for reading float values
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    """
    df = pd.read_csv(path, sep='\t', engine='c', na_values=['na', '-', ''],
        dtype=dtype, converters={'NAME': str})
    df1 = pd.DataFrame(df.loc[:, 'NAME'])
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)
    df2 = df.drop('NAME', axis=1)
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(dtype)
    df_dg = pd.concat([df1, df2], axis=1)
    return df_dg
