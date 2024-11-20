def load_data_all(self, switch_drugs=False):
    df_all = self.df_response
    y_all = df_all['GROWTH'].values
    x_all_list = []
    for fea in self.cell_features:
        df_cell = getattr(self, self.cell_df_dict[fea])
        df_x_all = pd.merge(df_all[['CELLNAME']], df_cell, on='CELLNAME',
            how='left')
        x_all_list.append(df_x_all.drop(['CELLNAME'], axis=1).values)
    drugs = ['NSC1', 'NSC2']
    doses = ['pCONC1', 'pCONC2']
    if switch_drugs:
        drugs = ['NSC2', 'NSC1']
        doses = ['pCONC2', 'pCONC1']
    for drug in drugs:
        for fea in self.drug_features:
            df_drug = getattr(self, self.drug_df_dict[fea])
            df_x_all = pd.merge(df_all[[drug]], df_drug, left_on=drug,
                right_on='NSC', how='left')
            x_all_list.append(df_x_all.drop([drug, 'NSC'], axis=1).values)
    for dose in doses:
        x_all_list.append(df_all[dose].values)
    return x_all_list, y_all, df_all
