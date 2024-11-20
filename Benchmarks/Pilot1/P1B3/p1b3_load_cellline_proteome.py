def load_cellline_proteome(path, dtype, kinome_path=None, ncols=None,
    scaling='std'):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    path: string
        path to 'nci60_proteome_log2.transposed.tsv'
    dtype: numpy type
        precision (data type) for reading float values
    kinome_path: string or None (default None)
        path to 'nci60_kinome_log2.transposed.tsv'
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply

    """
    df = pd.read_csv(path, sep='\t', engine='c')
    df = df.set_index('CellLine')
    if kinome_path:
        df_k = pd.read_csv(kinome_path, sep='\t', engine='c')
        df_k = df_k.set_index('CellLine')
        df_k = df_k.add_suffix('.K')
        df = df.merge(df_k, left_index=True, right_index=True)
    index = df.index.map(lambda x: x.replace('.', ':'))
    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df = df.iloc[:, usecols]
    df = impute_and_scale(df, scaling)
    df = df.astype(dtype)
    df.index = index
    df.index.names = ['CELLNAME']
    df = df.reset_index()
    return df
