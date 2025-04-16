def load_top_cell_types(gParams, scaling='minmax'):
    n = gParams['top_k_types']
    df_cell = load_cell_rnaseq(use_landmark_genes=True, preprocess_rnaseq=
        'source_scale', scaling=scaling)
    df_type = load_cell_type(gParams)
    df = pd.merge(df_type, df_cell, on='Sample')
    type_counts = dict(df.type.value_counts().nlargest(n))
    df_top_types = df[df.type.isin(type_counts.keys())]
    return df_top_types
