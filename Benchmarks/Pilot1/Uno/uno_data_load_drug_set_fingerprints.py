def load_drug_set_fingerprints(drug_set='Combined_PubChem', ncols=None,
    usecols=None, scaling=None, imputing=None, add_prefix=False):
    fps = ['PFP', 'ECFP']
    usecols_all = usecols
    df_merged = None
    for fp in fps:
        path = get_file(DATA_URL + '{}_dragon7_{}.tsv'.format(drug_set, fp))
        df_cols = pd.read_csv(path, engine='c', sep='\t', nrows=0, skiprows
            =1, header=None)
        total = df_cols.shape[1] - 1
        if usecols_all is not None:
            usecols = [x.replace(fp + '.', '') for x in usecols_all]
            usecols = [int(x) for x in usecols if x.isdigit()]
            usecols = [x for x in usecols if x in df_cols.columns]
            if usecols[0] != 0:
                usecols = [0] + usecols
            df_cols = df_cols.loc[:, usecols]
        elif ncols and ncols < total:
            usecols = np.random.choice(total, size=ncols, replace=False)
            usecols = np.append([0], np.add(sorted(usecols), 1))
            df_cols = df_cols.iloc[:, usecols]
        dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
        df = pd.read_csv(path, engine='c', sep='\t', skiprows=1, header=
            None, usecols=usecols, dtype=dtype_dict)
        df.columns = ['{}.{}'.format(fp, x) for x in df.columns]
        col1 = '{}.0'.format(fp)
        df1 = pd.DataFrame(df.loc[:, col1])
        df1.rename(columns={col1: 'Drug'}, inplace=True)
        df2 = df.drop(col1, axis=1)
        if add_prefix:
            df2 = df2.add_prefix('dragon7.')
        df2 = impute_and_scale(df2, scaling, imputing, dropna=None)
        df = pd.concat([df1, df2], axis=1)
        df_merged = df if df_merged is None else df_merged.merge(df)
    return df_merged
