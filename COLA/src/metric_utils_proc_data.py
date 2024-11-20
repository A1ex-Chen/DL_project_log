def proc_data(df):
    for col in ['covariates', 'interventions']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    for col in ['p_xd', 'p_dy', 'p_xy']:
        df[col] = df[col].apply(lambda s: np.array(ast.literal_eval(s)))
    return df
