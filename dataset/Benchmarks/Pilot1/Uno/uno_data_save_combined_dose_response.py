def save_combined_dose_response():
    df1 = load_single_dose_response(combo_format=True, fraction=False)
    df2 = load_combo_dose_response(fraction=False)
    df = pd.concat([df1, df2])
    df.to_csv('combined_drug_growth', index=False, sep='\t')
