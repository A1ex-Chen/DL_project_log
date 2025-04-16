def load_drug_data(ncols=None, scaling='std', imputing='mean', dropna=None,
    add_prefix=True):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']
    df_desc = load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=
        ncols)
    df_fp = load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=ncols
        )
    df_desc = pd.merge(df_info[['ID', 'Drug']], df_desc, on='Drug').drop('Drug'
        , axis=1).rename(columns={'ID': 'Drug'})
    df_fp = pd.merge(df_info[['ID', 'Drug']], df_fp, on='Drug').drop('Drug',
        axis=1).rename(columns={'ID': 'Drug'})
    df_desc2 = load_drug_set_descriptors(drug_set='NCI60', usecols=df_desc.
        columns.tolist() if ncols else None)
    df_fp2 = load_drug_set_fingerprints(drug_set='NCI60', usecols=df_fp.
        columns.tolist() if ncols else None)
    df_desc = pd.concat([df_desc, df_desc2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_desc.loc[:, 'Drug'])
    df2 = df_desc.drop('Drug', axis=1)
    df2 = impute_and_scale(df2, scaling=scaling, imputing=imputing, dropna=
        dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_desc = pd.concat([df1, df2], axis=1)
    df_fp = pd.concat([df_fp, df_fp2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_fp.loc[:, 'Drug'])
    df2 = df_fp.drop('Drug', axis=1)
    df2 = impute_and_scale(df2, scaling=None, imputing=imputing, dropna=dropna)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df_fp = pd.concat([df1, df2], axis=1)
    logger.info('Loaded combined dragon7 drug descriptors: %s', df_desc.shape)
    logger.info('Loaded combined dragon7 drug fingerprints: %s', df_fp.shape)
    return df_desc, df_fp
