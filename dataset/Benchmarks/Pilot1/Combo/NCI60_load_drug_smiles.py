def load_drug_smiles():
    path = get_file(DATA_URL + 'ChemStructures_Consistent.smiles')
    df = global_cache.get(path)
    if df is None:
        df = pd.read_csv(path, sep='\t', engine='c', dtype={'nsc_id': object})
        df = df.rename(columns={'nsc_id': 'NSC'})
        global_cache[path] = df
    return df
