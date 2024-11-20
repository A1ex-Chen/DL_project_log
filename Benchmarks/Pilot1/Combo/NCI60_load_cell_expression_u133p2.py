def load_cell_expression_u133p2(ncols=None, scaling='std', add_prefix=True,
    use_landmark_genes=False):
    """Load U133_Plus2 cell line expression data prepared by Judith,
        sub-select columns of gene expression randomly if specificed,
        scale the selected data and return a pandas dataframe.

    Parameters
    ----------
    ncols : int or None
        number of columns (gene expression) to randomly subselect (default None : use all data)
    scaling : 'maxabs' [-1,1], 'minmax' [0,1], 'std', or None, optional (default 'std')
        type of scaling to apply
    add_prefix: True or False
        add feature namespace prefix
    use_landmark_genes: True or False
        only use LINCS landmark genes (L1000)
    """
    path = get_file(DATA_URL + 'GSE32474_U133Plus2_GCRMA_gene_median.txt')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c')
        global_cache[path] = df
    if use_landmark_genes:
        lincs_path = get_file(DATA_URL + 'lincs1000.tsv')
        df_l1000 = pd.read_csv(lincs_path, sep='\t')
        cols = sorted([x for x in df_l1000['symbol'] if x in df.columns])
        df = df[['CELLNAME'] + cols]
    df1 = df['CELLNAME']
    df1 = df1.map(lambda x: x.replace(':', '.'))
    df2 = df.drop('CELLNAME', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('expr.')
    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)
    return df
