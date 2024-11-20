def lookup(df, sample, drug1, drug2=None, value=None):
    drug2 = drug2 or drug1
    df_result = df[(df['Sample'] == sample) & (df['Drug1'] == drug1) & (df[
        'Drug2'] == drug2)]
    if df_result.empty:
        df_result = df[(df['Sample'] == sample) & (df['Drug1'] == drug2) &
            (df['Drug2'] == drug1)]
    if value:
        if df_result.empty:
            return 1.0
        else:
            return df_result[value].iloc[0]
    else:
        return df_result
