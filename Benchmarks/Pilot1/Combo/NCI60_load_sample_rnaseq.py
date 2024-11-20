def load_sample_rnaseq(ncols=None, scaling='std', add_prefix=True,
    use_landmark_genes=False, preprocess_rnaseq=None, sample_set='NCI60'):
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
    if sample_set == 'RTS':
        df_ids = pd.read_table(get_file(DATA_URL + 'RTS_PDM_samples'))
        df = df.merge(df_ids, on='Sample').reset_index(drop=True)
    else:
        df = df[df['Sample'].str.startswith(sample_set)].reset_index(drop=True)
    df1 = df['Sample']
    df2 = df.drop('Sample', axis=1)
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
