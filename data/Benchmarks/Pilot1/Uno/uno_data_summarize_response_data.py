def summarize_response_data(df, target=None):
    target = target or 'Growth'
    df_sum = df.groupby('Source').agg({target: 'count', 'Sample': 'nunique',
        'Drug1': 'nunique', 'Drug2': 'nunique'})
    if 'Dose1' in df_sum:
        df_sum['MedianDose'] = df.groupby('Source').agg({'Dose1': 'median'})
    return df_sum
