def load_drug_fingerprints(ncols=None, scaling='std', imputing='mean',
    dropna=None, add_prefix=True, feature_subset=None):
    df_info = load_drug_info()
    df_info['Drug'] = df_info['PUBCHEM']
    df_fp = load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=ncols
        )
    df_fp = pd.merge(df_info[['ID', 'Drug']], df_fp, on='Drug').drop('Drug',
        axis=1).rename(columns={'ID': 'Drug'})
    df_fp2 = load_drug_set_fingerprints(drug_set='NCI60', usecols=df_fp.
        columns.tolist() if ncols else None)
    df_fp = pd.concat([df_fp, df_fp2]).reset_index(drop=True)
    df1 = pd.DataFrame(df_fp.loc[:, 'Drug'])
    df2 = df_fp.drop('Drug', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    if feature_subset:
        df2 = df2[[x for x in df2.columns if x in feature_subset]]
    df2 = impute_and_scale(df2, scaling=None, imputing=imputing, dropna=dropna)
    df_fp = pd.concat([df1, df2], axis=1)
    logger.info('Loaded combined dragon7 drug fingerprints: %s', df_fp.shape)
    return df_fp
