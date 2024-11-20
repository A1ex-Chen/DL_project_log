def select_drugs_with_response_range(df_response, lower=0, upper=0, span=0,
    lower_median=None, upper_median=None):
    df = df_response.groupby(['Drug1', 'Sample'])['Growth'].agg(['min',
        'max', 'median'])
    df['span'] = df['max'].clip(lower=-1, upper=1) - df['min'].clip(lower=-
        1, upper=1)
    df = df.groupby('Drug1').mean().reset_index().rename(columns={'Drug1':
        'Drug'})
    mask = (df['min'] <= lower) & (df['max'] >= upper) & (df['span'] >= span)
    if lower_median:
        mask &= df['median'] >= lower_median
    if upper_median:
        mask &= df['median'] <= upper_median
    df_sub = df[mask]
    return df_sub
