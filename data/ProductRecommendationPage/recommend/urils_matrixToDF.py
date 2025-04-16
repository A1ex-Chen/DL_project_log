def matrixToDF(df_matrix):
    df = df_matrix.stack().reset_index()
    df = df.rename(columns={(0): 'rating'})
    return df
