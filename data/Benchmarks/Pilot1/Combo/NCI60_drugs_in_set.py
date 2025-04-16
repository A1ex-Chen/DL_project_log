def drugs_in_set(set_name):
    path = get_file(DATA_URL + 'NCI60_drug_sets.tsv')
    df = pd.read_csv(path, sep='\t', engine='c')
    drugs = df[df['Drug_Set'] == set_name].iloc[0][1].split(',')
    return drugs
