def make_group_from_columns(df, groupcols):
    return df[groupcols].astype(str).sum(axis=1).values
