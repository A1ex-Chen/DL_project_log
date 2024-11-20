def all_drugs():
    df = load_dose_response()
    return df['NSC'].drop_duplicates().tolist()
