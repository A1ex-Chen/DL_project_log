def load_single_dose_response(combo_format=False, fraction=True):
    path = get_file(DATA_URL + 'rescaled_combined_single_drug_growth')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c', na_values=['na', '-',
            ''], dtype={'SOURCE': str, 'DRUG_ID': str, 'CELLNAME': str,
            'CONCUNIT': str, 'LOG_CONCENTRATION': np.float32, 'EXPID': str,
            'GROWTH': np.float32})
        global_cache[path] = df
    df['DOSE'] = -df['LOG_CONCENTRATION']
    df = df.rename(columns={'CELLNAME': 'CELL', 'DRUG_ID': 'DRUG', 'EXPID':
        'STUDY'})
    df = df[['SOURCE', 'CELL', 'DRUG', 'DOSE', 'GROWTH', 'STUDY']]
    if fraction:
        df['GROWTH'] /= 100
    if combo_format:
        df = df.rename(columns={'DRUG': 'DRUG1', 'DOSE': 'DOSE1'})
        df['DRUG2'] = np.nan
        df['DOSE2'] = np.nan
        df['DRUG2'] = df['DRUG2'].astype(object)
        df['DOSE2'] = df['DOSE2'].astype(np.float32)
        df = df[['SOURCE', 'CELL', 'DRUG1', 'DOSE1', 'DRUG2', 'DOSE2',
            'GROWTH', 'STUDY']]
    return df
