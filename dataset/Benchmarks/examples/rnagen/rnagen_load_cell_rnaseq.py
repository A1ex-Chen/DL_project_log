def load_cell_rnaseq(ncols=None, scaling='std', imputing='mean', add_prefix
    =True, use_landmark_genes=False, use_filtered_genes=False,
    feature_subset=None, preprocess_rnaseq=None, embed_feature_source=False,
    sample_set=None, index_by_sample=False):
    if use_landmark_genes:
        filename = 'combined_rnaseq_data_lincs1000'
    elif use_filtered_genes:
        filename = 'combined_rnaseq_data_filtered'
    else:
        filename = 'combined_rnaseq_data'
    if preprocess_rnaseq and preprocess_rnaseq != 'none':
        filename += '_' + preprocess_rnaseq
    path = get_file(DATA_URL + filename)
    df_cols = pd.read_csv(path, engine='c', sep='\t', nrows=0)
    total = df_cols.shape[1] - 1
    if 'Cancer_type_id' in df_cols.columns:
        total -= 1
    usecols = None
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        usecols = np.append([0], np.add(sorted(usecols), 2))
        df_cols = df_cols.iloc[:, usecols]
    if feature_subset:

        def with_prefix(x):
            return 'rnaseq.' + x if add_prefix else x
        usecols = [0] + [i for i, c in enumerate(df_cols.columns) if 
            with_prefix(c) in feature_subset]
        df_cols = df_cols.iloc[:, usecols]
    dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
    df = pd.read_csv(path, engine='c', sep='\t', usecols=usecols, dtype=
        dtype_dict)
    if 'Cancer_type_id' in df.columns:
        df.drop('Cancer_type_id', axis=1, inplace=True)
    prefixes = df['Sample'].str.extract('^([^.]*)', expand=False).rename(
        'Source')
    sources = prefixes.drop_duplicates().reset_index(drop=True)
    df_source = pd.get_dummies(sources, prefix='rnaseq.source', prefix_sep='.')
    df_source = pd.concat([sources, df_source], axis=1)
    df1 = df['Sample']
    if embed_feature_source:
        df_sample_source = pd.concat([df1, prefixes], axis=1)
        df1 = df_sample_source.merge(df_source, on='Source', how='left').drop(
            'Source', axis=1)
        logger.info(
            'Embedding RNAseq data source into features: %d additional columns'
            , df1.shape[1] - 1)
    df2 = df.drop('Sample', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('rnaseq.')
    if scaling:
        df2 = impute_and_scale(df2, scaling, imputing)
    df = pd.concat([df1, df2], axis=1)
    if sample_set:
        chosen = df['Sample'].str.startswith(sample_set)
        df = df[chosen].reset_index(drop=True)
    if index_by_sample:
        df = df.set_index('Sample')
    logger.info('Loaded combined RNAseq data: %s', df.shape)
    return df
