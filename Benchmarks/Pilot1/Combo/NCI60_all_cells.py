def all_cells():
    df = load_dose_response()
    return df['CELLNAME'].drop_duplicates().tolist()
