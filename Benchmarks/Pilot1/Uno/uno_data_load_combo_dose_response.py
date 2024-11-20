def load_combo_dose_response(fraction=True):
    path = get_file(DATA_URL + 'ComboDrugGrowth_Nov2017.csv')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep=',', engine='c', na_values=['na', '-',
            ''], usecols=['CELLNAME', 'NSC1', 'CONC1', 'NSC2', 'CONC2',
            'PERCENTGROWTH', 'VALID', 'SCREENER', 'STUDY'], dtype={
            'CELLNAME': str, 'NSC1': str, 'NSC2': str, 'CONC1': np.float32,
            'CONC2': np.float32, 'PERCENTGROWTH': np.float32, 'VALID': str,
            'SCREENER': str, 'STUDY': str}, error_bad_lines=False,
            warn_bad_lines=True)
        global_cache[path] = df
    df = df[df['VALID'] == 'Y']
    df['SOURCE'] = 'ALMANAC.' + df['SCREENER']
    cellmap_path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.txt')
    df_cellmap = pd.read_csv(cellmap_path, sep='\t')
    df_cellmap.set_index('Name', inplace=True)
    cellmap = df_cellmap[['NCI60.ID']].to_dict()['NCI60.ID']
    df['CELL'] = df['CELLNAME'].map(lambda x: cellmap[x])
    df['DOSE1'] = -np.log10(df['CONC1'])
    df['DOSE2'] = -np.log10(df['CONC2'])
    df['DRUG1'] = 'NSC.' + df['NSC1']
    df['DRUG2'] = 'NSC.' + df['NSC2']
    if fraction:
        df['GROWTH'] = df['PERCENTGROWTH'] / 100
    else:
        df['GROWTH'] = df['PERCENTGROWTH']
    df = df[['SOURCE', 'CELL', 'DRUG1', 'DOSE1', 'DRUG2', 'DOSE2', 'GROWTH',
        'STUDY']]
    return df
