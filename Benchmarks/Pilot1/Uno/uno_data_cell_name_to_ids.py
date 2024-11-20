def cell_name_to_ids(name, source=None):
    path = get_file(DATA_URL + 'NCI60_CELLNAME_to_Combo.txt')
    df1 = pd.read_csv(path, sep='\t')
    hits1 = lookup(df1, name, 'NCI60.ID', ['NCI60.ID', 'CELLNAME', 'Name'],
        match='contains')
    path = get_file(DATA_URL + 'cl_mapping')
    df2 = pd.read_csv(path, sep='\t', header=None)
    hits2 = lookup(df2, name, [0, 1], [0, 1], match='contains')
    hits = hits1 + hits2
    if source:
        hits = [x for x in hits if x.startswith(source.upper() + '.')]
    return hits
