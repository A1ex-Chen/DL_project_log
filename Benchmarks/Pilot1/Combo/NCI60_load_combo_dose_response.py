def load_combo_dose_response(response_url=None, fraction=False,
    use_combo_score=False, exclude_cells=[], exclude_drugs=[]):
    """Load cell line response to pairs of drugs, sub-select response for a specific
        drug log concentration range and return a pandas dataframe.

    Parameters
    ----------
    fraction: bool (default False)
        divide growth percentage by 100
    use_combo_score: bool (default False)
        return combination score in place of percent growth (stored in 'GROWTH' column)
    """
    response_url = response_url or DATA_URL + 'ComboDrugGrowth_Nov2017.csv'
    path = get_file(response_url)
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, usecols=['CELLNAME', 'NSC1', 'CONC1', 'NSC2',
            'CONC2', 'PERCENTGROWTH', 'VALID', 'SCORE', 'SCREENER', 'STUDY'
            ], na_values=['na', '-', ''], dtype={'NSC1': object, 'NSC2':
            object, 'CONC1': object, 'CONC2': object, 'PERCENTGROWTH': str,
            'SCORE': str}, engine='c', error_bad_lines=False,
            warn_bad_lines=True)
        global_cache[path] = df
    df = df[df['VALID'] == 'Y']
    df = df[['CELLNAME', 'NSC1', 'NSC2', 'CONC1', 'CONC2', 'PERCENTGROWTH',
        'SCORE']]
    exclude_cells = [x.split('.')[-1] for x in exclude_cells]
    exclude_drugs = [x.split('.')[-1] for x in exclude_drugs]
    df = df[~df['CELLNAME'].isin(exclude_cells) & ~df['NSC1'].isin(
        exclude_drugs) & ~df['NSC2'].isin(exclude_drugs)]
    df['PERCENTGROWTH'] = df['PERCENTGROWTH'].astype(np.float32)
    df['SCORE'] = df['SCORE'].astype(np.float32)
    df['NSC2'] = df['NSC2'].fillna(df['NSC1'])
    df['CONC2'] = df['CONC2'].fillna(df['CONC1'])
    df['SCORE'] = df['SCORE'].fillna(0)
    cellmap_path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.txt')
    df_cellmap = pd.read_csv(cellmap_path, sep='\t')
    df_cellmap.set_index('Name', inplace=True)
    cellmap = df_cellmap[['CELLNAME']].to_dict()['CELLNAME']
    df['CELLNAME'] = df['CELLNAME'].map(lambda x: cellmap[x])
    df_mean = df.groupby(['CELLNAME', 'NSC1', 'NSC2', 'CONC1', 'CONC2']).mean()
    df_mean = df_mean.reset_index()
    df_mean['CONC1'] = -np.log10(df_mean['CONC1'].astype(np.float32))
    df_mean['CONC2'] = -np.log10(df_mean['CONC2'].astype(np.float32))
    df = df_mean.rename(columns={'PERCENTGROWTH': 'GROWTH', 'CONC1':
        'pCONC1', 'CONC2': 'pCONC2'})
    if fraction:
        df['GROWTH'] /= 100
    return df
