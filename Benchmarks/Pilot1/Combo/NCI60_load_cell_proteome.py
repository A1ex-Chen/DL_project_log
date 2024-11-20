def load_cell_proteome(ncols=None, scaling='std', add_prefix=True):
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
    path1 = get_file(P1B3_URL + 'nci60_proteome_log2.transposed.tsv')
    path2 = get_file(P1B3_URL + 'nci60_kinome_log2.transposed.tsv')
    df = global_cache.get(path1)
    if df is None:
        df = pd.read_csv(path1, sep='\t', engine='c')
        global_cache[path1] = df
    df_k = global_cache.get(path2)
    if df_k is None:
        df_k = pd.read_csv(path2, sep='\t', engine='c')
        global_cache[path2] = df_k
    df = df.set_index('CellLine')
    df_k = df_k.set_index('CellLine')
    if add_prefix:
        df = df.add_prefix('prot.')
        df_k = df_k.add_prefix('kino.')
    else:
        df_k = df_k.add_suffix('.K')
    df = df.merge(df_k, left_index=True, right_index=True)
    index = df.index
    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df = df.iloc[:, usecols]
    df = impute_and_scale(df, scaling)
    df = df.astype(np.float32)
    df.index = index
    df.index.names = ['CELLNAME']
    df = df.reset_index()
    return df
