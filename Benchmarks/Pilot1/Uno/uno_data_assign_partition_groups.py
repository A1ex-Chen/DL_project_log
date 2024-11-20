def assign_partition_groups(df, partition_by='drug_pair'):
    if partition_by == 'cell':
        group = df['Sample']
    elif partition_by == 'drug_pair':
        df_info = load_drug_info()
        id_dict = df_info[['ID', 'PUBCHEM']].drop_duplicates(['ID']).set_index(
            'ID').iloc[:, 0]
        group = df['Drug1'].copy()
        group[df['Drug2'].notnull() & (df['Drug1'] <= df['Drug2'])] = df[
            'Drug1'] + ',' + df['Drug2']
        group[df['Drug2'].notnull() & (df['Drug1'] > df['Drug2'])] = df['Drug2'
            ] + ',' + df['Drug1']
        group2 = group.map(id_dict)
        mapped = group2.notnull()
        group[mapped] = group2[mapped]
    elif partition_by == 'index':
        group = df.reset_index()['index']
    logger.info('Grouped response data by %s: %d groups', partition_by,
        group.nunique())
    return group
