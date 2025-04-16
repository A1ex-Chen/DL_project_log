def load_cell_mirna(ncols=None, scaling='std', add_prefix=True):
    """Load cell line microRNA data, sub-select columns randomly if
        specificed, scale the selected data and return a pandas
        dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """
    path = get_file(P1B3_URL + 'RNA__microRNA_OSU_V3_chip_log2.transposed.txt')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c', na_values=['na', '-', ''])
        global_cache[path] = df
    df1 = df['CellLine']
    df1.name = 'CELLNAME'
    df2 = df.drop('CellLine', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('mRNA.')
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)
    return df
