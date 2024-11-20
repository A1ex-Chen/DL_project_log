def categorize_dataframe(df, ycol='0', bins=5, cutoffs=None, verbose=False):
    if ycol.isdigit():
        ycol = df.columns[int(ycol)]
    y = df.loc[:, ycol].values
    classes = discretize(y, bins, cutoffs, verbose)
    df.iloc[:, 0] = classes
    return df
