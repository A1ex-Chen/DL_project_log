def lookup(df, query, ret, keys, match='match'):
    mask = pd.Series(False, index=range(df.shape[0]))
    for key in keys:
        if match == 'contains':
            mask |= df[key].str.contains(query.upper(), case=False)
        else:
            mask |= df[key].str.upper() == query.upper()
    return list(set(df[mask][ret].values.flatten().tolist()))
