def load_aggregated_single_response(target='AUC', min_r2_fit=0.3,
    max_ec50_se=3, combo_format=False, rename=True):
    path = get_file(DATA_URL + 'combined_single_response_agg')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, engine='c', sep='\t', dtype={'SOURCE': str,
            'CELL': str, 'DRUG': str, 'STUDY': str, 'AUC': np.float32,
            'IC50': np.float32, 'EC50': np.float32, 'EC50se': np.float32,
            'R2fit': np.float32, 'Einf': np.float32, 'HS': np.float32,
            'AAC1': np.float32, 'AUC1': np.float32, 'DSS1': np.float32})
        global_cache[path] = df
    total = len(df)
    df = df[(df['R2fit'] >= min_r2_fit) & (df['EC50se'] <= max_ec50_se)]
    df = df[['SOURCE', 'CELL', 'DRUG', target, 'STUDY']]
    df = df[~df[target].isnull()]
    logger.info(
        'Loaded %d dose indepdendent response samples (filtered by EC50se <= %f & R2fit >=%f from a total of %d).'
        , len(df), max_ec50_se, min_r2_fit, total)
    if combo_format:
        df = df.rename(columns={'DRUG': 'DRUG1'})
        df['DRUG2'] = np.nan
        df['DRUG2'] = df['DRUG2'].astype(object)
        df = df[['SOURCE', 'CELL', 'DRUG1', 'DRUG2', target, 'STUDY']]
        if rename:
            df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                'DRUG1': 'Drug1', 'DRUG2': 'Drug2', 'STUDY': 'Study'})
    elif rename:
        df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
            'DRUG': 'Drug', 'STUDY': 'Study'})
    return df
