def load_drug_set_descriptors(drug_set='ALMANAC', ncols=None, scaling='std',
    add_prefix=True):
    if drug_set == 'ALMANAC':
        path = get_file(DATA_URL + 'ALMANAC_drug_descriptors_dragon7.txt')
    elif drug_set == 'GDSC':
        path = get_file(DATA_URL + 'GDSC_PubChemCID_drug_descriptors_dragon7')
    elif drug_set == 'NCI_IOA_AOA':
        path = get_file(DATA_URL + 'NCI_IOA_AOA_drug_descriptors_dragon7')
    elif drug_set == 'RTS':
        path = get_file(DATA_URL + 'RTS_drug_descriptors_dragon7')
    elif drug_set == 'pan':
        path = get_file(DATA_URL + 'pan_drugs_dragon7_descriptors.tsv')
    else:
        raise Exception('Drug set {} not supported!'.format(drug_set))
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c', na_values=['na', '-', ''])
        global_cache[path] = df
    df1 = pd.DataFrame(df.loc[:, 'NAME'])
    df1.rename(columns={'NAME': 'Drug'}, inplace=True)
    df2 = df.drop('NAME', axis=1)
    if add_prefix:
        df2 = df2.add_prefix('dragon7.')
    total = df2.shape[1]
    if ncols and ncols < total:
        usecols = np.random.choice(total, size=ncols, replace=False)
        df2 = df2.iloc[:, usecols]
        keepcols = None
    else:
        train_ref = load_drug_descriptors(add_prefix=add_prefix)
        keepcols = train_ref.columns[1:]
    df2 = impute_and_scale(df2, scaling, keepcols=keepcols)
    df2 = df2.astype(np.float32)
    df_dg = pd.concat([df1, df2], axis=1)
    return df_dg
