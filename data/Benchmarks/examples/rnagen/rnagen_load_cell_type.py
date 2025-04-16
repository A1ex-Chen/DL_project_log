def load_cell_type():
    path = get_file(DATA_URL + 'combined_cancer_types')
    df = pd.read_csv(path, engine='c', sep='\t', header=None)
    df.columns = ['Sample', 'type']
    return df
