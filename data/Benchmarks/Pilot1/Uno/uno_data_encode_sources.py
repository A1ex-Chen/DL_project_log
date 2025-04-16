def encode_sources(sources):
    df = pd.get_dummies(sources, prefix='source', prefix_sep='.')
    df['Source'] = sources
    source_l1 = df['Source'].str.extract('^(\\S+)\\.', expand=False)
    df1 = pd.get_dummies(source_l1, prefix='source.L1', prefix_sep='.')
    df = pd.concat([df1, df], axis=1)
    df = df.set_index('Source').reset_index()
    return df
