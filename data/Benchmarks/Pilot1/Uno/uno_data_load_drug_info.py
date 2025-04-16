def load_drug_info():
    path = get_file(DATA_URL + 'drug_info')
    df = pd.read_csv(path, sep='\t', dtype=object)
    df['PUBCHEM'] = 'PubChem.CID.' + df['PUBCHEM']
    return df
