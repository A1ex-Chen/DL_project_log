def drug_name_to_ids(name, source=None):
    df1 = load_drug_info()
    path = get_file(DATA_URL + 'NCI_IOA_AOA_drugs')
    df2 = pd.read_csv(path, sep='\t', dtype=str)
    df2['NSC'] = 'NSC.' + df2['NSC']
    hits1 = lookup(df1, name, 'ID', ['ID', 'NAME', 'CLEAN_NAME', 'PUBCHEM'])
    hits2 = lookup(df2, name, 'NSC', ['NSC', 'Generic Name', 'Preffered Name'])
    hits = hits1 + hits2
    if source:
        hits = [x for x in hits if x.startswith(source.upper() + '.')]
    return hits
