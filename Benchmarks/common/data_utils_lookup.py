def lookup(df, query, ret, keys, match='match'):
    """Dataframe lookup.

    Parameters
    ----------
    df : pandas dataframe
        dataframe for retrieving values.
    query : string
        String for searching.
    ret : int/string or list
        Names or indices of columns to be returned.
    keys : list
        List of strings or integers specifying the names or
        indices of columns to look into.
    match : string
        String describing strategy for matching keys to query.

    Return
    ----------
    Returns a list of the values in the dataframe whose columns match
    the specified query and have been selected to be returned.

    """
    mask = pd.Series(False, index=range(df.shape[0]))
    for key in keys:
        if match == 'contains':
            mask |= df[key].str.contains(query.upper(), case=False)
        else:
            mask |= df[key].str.upper() == query.upper()
    return list(set(df[mask][ret].values.flatten().tolist()))
