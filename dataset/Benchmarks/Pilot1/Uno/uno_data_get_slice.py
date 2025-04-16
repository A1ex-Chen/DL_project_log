def get_slice(self, size=None, contiguous=True, single=False, dataframe=
    False, partial_index=None):
    size = size or self.size
    single = single or self.data.agg_dose
    target = self.data.agg_dose or 'Growth'
    if partial_index is not None:
        index = partial_index
    else:
        index = list(islice(self.index_cycle, size))
    df_orig = self.data.df_response.iloc[index, :]
    df = df_orig.copy()
    if not single:
        df['Swap'] = np.random.choice([True, False], df.shape[0])
        swap = df_orig['Drug2'].notnull() & df['Swap']
        df.loc[swap, 'Drug1'] = df_orig.loc[swap, 'Drug2']
        df.loc[swap, 'Drug2'] = df_orig.loc[swap, 'Drug1']
        if not self.data.agg_dose:
            df['DoseSplit'] = np.random.uniform(0.001, 0.999, df.shape[0])
            df.loc[swap, 'Dose1'] = df_orig.loc[swap, 'Dose2']
            df.loc[swap, 'Dose2'] = df_orig.loc[swap, 'Dose1']
    split = df_orig['Drug2'].isnull()
    if not single:
        df.loc[split, 'Drug2'] = df_orig.loc[split, 'Drug1']
        if not self.data.agg_dose:
            df.loc[split, 'Dose1'] = df_orig.loc[split, 'Dose1'] - np.log10(df
                .loc[split, 'DoseSplit'])
            df.loc[split, 'Dose2'] = df_orig.loc[split, 'Dose1'] - np.log10(
                1 - df.loc[split, 'DoseSplit'])
    if dataframe:
        cols = [target, 'Sample', 'Drug1', 'Drug2'] if not single else [target,
            'Sample', 'Drug1']
        y = df[cols].reset_index(drop=True)
    else:
        y = values_or_dataframe(df[target], contiguous, dataframe)
    x_list = []
    if not self.data.agg_dose:
        doses = ['Dose1', 'Dose2'] if not single else ['Dose1']
        for dose in doses:
            x = values_or_dataframe(df[[dose]].reset_index(drop=True),
                contiguous, dataframe)
            x_list.append(x)
    if self.data.encode_response_source:
        df_x = pd.merge(df[['Source']], self.data.df_source, on='Source',
            how='left')
        df_x.drop(['Source'], axis=1, inplace=True)
        x = values_or_dataframe(df_x, contiguous, dataframe)
        x_list.append(x)
    for fea in self.data.cell_features:
        df_cell = getattr(self.data, self.data.cell_df_dict[fea])
        df_x = pd.merge(df[['Sample']], df_cell, on='Sample', how='left')
        df_x.drop(['Sample'], axis=1, inplace=True)
        x = values_or_dataframe(df_x, contiguous, dataframe)
        x_list.append(x)
    drugs = ['Drug1', 'Drug2'] if not single else ['Drug1']
    for drug in drugs:
        for fea in self.data.drug_features:
            df_drug = getattr(self.data, self.data.drug_df_dict[fea])
            df_x = pd.merge(df[[drug]], df_drug, left_on=drug, right_on=
                'Drug', how='left')
            df_x.drop([drug, 'Drug'], axis=1, inplace=True)
            if dataframe and not single:
                df_x = df_x.add_prefix(drug + '.')
            x = values_or_dataframe(df_x, contiguous, dataframe)
            x_list.append(x)
    return x_list, y
