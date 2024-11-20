def load_drug_set_descriptors(drug_set='Combined_PubChem', ncols=None,
    usecols=None, scaling=None, imputing=None, add_prefix=False):
    path = get_file(DATA_URL + '{}_dragon7_descriptors.tsv'.format(drug_set))
    df_cols = pd.read_csv(path, engine='c', sep='\t', nrows=0)
    total = df_cols.shape[1] - 1
    if usecols is not None:
        usecols = [x for x in usecols if x in df_cols.columns]
        if usecols[0] != 'NAME':
            usecols = ['NAME'] + usecols
        df_cols = df_cols.loc[:, usecols]
    elif ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        usecols = np.append([0], np.add(sorted(usecols), 1))
        df_cols = df_cols.iloc[:, usecols]
    dtype_dict = dict((x, np.float32) for x in df_cols.columns[1:])
    df = pd.read_csv(path, engine='c', sep='\t', usecols=usecols, dtype=
        dtype_dict, na_values=['na', '-', ''])
    df1 = pd.DataFrame(df.loc[:, 'NAME'])
    df1.rename(columns={'NAME': 'Drug'}, inplace=True)
    df2 = df.drop('NAME', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    df2 = impute_and_scale(df2, scaling, imputing, dropna=None)
    df = pd.concat([df1, df2], axis=1)
    return df
