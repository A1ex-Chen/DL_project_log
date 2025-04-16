def values_or_dataframe(df, contiguous=False, dataframe=False):
    if dataframe:
        return df
    mat = df.values
    if contiguous:
        mat = np.ascontiguousarray(mat)
    return mat
