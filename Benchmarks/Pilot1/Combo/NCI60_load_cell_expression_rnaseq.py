def load_cell_expression_rnaseq(ncols=None, scaling='std', add_prefix=True,
    use_landmark_genes=False, preprocess_rnaseq=None):
    if use_landmark_genes:
        filename = 'combined_rnaseq_data_lincs1000'
    else:
        filename = 'combined_rnaseq_data'
    if preprocess_rnaseq and preprocess_rnaseq != 'none':
        scaling = None
        filename += '_' + preprocess_rnaseq
    path = get_file(DATA_URL + filename)
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c')
        global_cache[path] = df
    df = df[df['Sample'].str.startswith('NCI60')].reset_index(drop=True)
    cellmap_path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.new.txt')
    df_cellmap = pd.read_csv(cellmap_path, sep='\t')
    df_cellmap.set_index('NCI60.ID', inplace=True)
    cellmap = df_cellmap[['CELLNAME']].to_dict()['CELLNAME']
    df = df.rename(columns={'Sample': 'CELLNAME'})
    df['CELLNAME'] = df['CELLNAME'].map(lambda x: cellmap[x])
    df1 = df['CELLNAME']
    df1 = df1.map(lambda x: x.replace(':', '.'))
    df2 = df.drop('CELLNAME', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('rnaseq.')
    total = df.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
    df2 = impute_and_scale(df2, scaling)
    df2 = df2.astype(np.float32)
    df = pd.concat([df1, df2], axis=1)
    return df
