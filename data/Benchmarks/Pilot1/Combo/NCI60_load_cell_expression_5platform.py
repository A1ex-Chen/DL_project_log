def load_cell_expression_5platform(ncols=None, scaling='std', add_prefix=
    True, use_landmark_genes=False):
    """Load 5-platform averaged cell line expression data, sub-select
        columns of gene expression randomly if specificed, scale the
        selected data and return a pandas dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    use_landmark_genes: True or False
        only use LINCS1000 landmark genes
    """
    path = get_file(P1B3_URL +
        'RNA_5_Platform_Gene_Transcript_Averaged_intensities.transposed.txt')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c', na_values=['na', '-', ''])
        global_cache[path] = df
    if use_landmark_genes:
        lincs_path = get_file(DATA_URL + 'lincs1000.tsv')
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        cols = sorted([x for x in df_l1000['symbol'] if x in df.columns])
        df = df[['CellLine'] + cols]
    df1 = df['CellLine']
    df1.name = 'CELLNAME'
    df2 = df.drop('CellLine', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('expr_5p.')
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)
    return df
