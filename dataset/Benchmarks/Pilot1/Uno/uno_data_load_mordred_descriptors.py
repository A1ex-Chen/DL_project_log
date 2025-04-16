def load_mordred_descriptors(ncols=None, scaling='std', imputing='mean',
    dropna=None, add_prefix=True, feature_subset=None):
    path = get_file(DATA_URL + 'extended_combined_mordred_descriptors')
    df = pd.read_csv(path, engine='c', sep='\t', na_values=['na', '-', ''])
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(np.float32)
    df1 = pd.DataFrame(df.loc[:, 'DRUG'])
    df1.rename(columns={'DRUG': 'Drug'}, inplace=True)
    df2 = df.drop('DRUG', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('mordred.')
    df2 = impute_and_scale(df2, scaling, imputing)
    df_desc = pd.concat([df1, df2], axis=1)
    df1 = pd.DataFrame(df_desc.loc[:, 'Drug'])
    df2 = df_desc.drop('Drug', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('mordred.')
    if feature_subset:
        df2 = df2[[x for x in df2.columns if x in feature_subset]]
    df2 = impute_and_scale(df2, scaling=scaling, imputing=imputing, dropna=
        dropna)
    df_desc = pd.concat([df1, df2], axis=1)
    logger.info('Loaded Mordred drug descriptors: %s', df_desc.shape)
    return df_desc
