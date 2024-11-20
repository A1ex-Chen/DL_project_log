def load_drug_descriptors_old(ncols=None, scaling='std', add_prefix=True):
    """Load drug descriptor data, sub-select columns of drugs descriptors
        randomly if specificed, impute and scale the selected data, and return a
        pandas dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns (drugs descriptors) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    """
    path = get_file(P1B3_URL + 'descriptors.2D-NSC.5dose.filtered.txt')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c', na_values=['na', '-',
            ''], dtype=np.float32)
        global_cache[path] = df
    df1 = pd.DataFrame(df.loc[:, 'NAME'].astype(int).astype(str))
    df1.rename(columns={'NAME': 'NSC'}, inplace=True)
    df2 = df.drop('NAME', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df_dg = pd.concat([df1, df2], axis=1)
    return df_dg
