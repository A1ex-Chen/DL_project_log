def load_cell_metadata():
    path = get_file(DATA_URL + 'cl_metadata')
    df = pd.read_csv(path, sep='\t')
    return df
