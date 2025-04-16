def load_drug_descriptors(ncols=None, scaling='std', imputing='mean',
    dropna=None, add_prefix=True, feature_subset=None):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']
    df_desc = load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=
        ncols)
    df_desc = pd.merge(df_info[['ID', 'Drug']], df_desc, on='Drug').drop('Drug'
        , axis=1).rename(columns={'ID': 'Drug'})
    df_desc2 = load_drug_set_descriptors(drug_set='NCI60', usecols=df_desc.
        columns.tolist() if ncols else None)
    df_desc = pd.concat([df_desc, df_desc2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_desc.loc[:, 'Drug'])
    df2 = df_desc.drop('Drug', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    if feature_subset:
        df2 = df2[[x for x in df2.columns if x in feature_subset]]
    df2 = impute_and_scale(df2, scaling=scaling, imputing=imputing, dropna=
        dropna)
    df_desc = pd.concat([df1, df2], axis=1)
    logger.info('Loaded combined dragon7 drug descriptors: %s', df_desc.shape)
    return df_desc
