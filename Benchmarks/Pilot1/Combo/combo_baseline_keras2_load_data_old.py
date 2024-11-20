def load_data_old(self):
    df_train = self.df_response.iloc[:self.n_train, :]
    df_val = self.df_response.iloc[self.n_train:, :]
    y_train = df_train['GROWTH'].values
    y_val = df_val['GROWTH'].values
    x_train_list = []
    x_val_list = []
    for fea in self.cell_features:
        df_cell = getattr(self, self.cell_df_dict[fea])
        df_x_train = pd.merge(df_train[['CELLNAME']], df_cell, on=
            'CELLNAME', how='left')
        df_x_val = pd.merge(df_val[['CELLNAME']], df_cell, on='CELLNAME',
            how='left')
        x_train_list.append(df_x_train.drop(['CELLNAME'], axis=1).values)
        x_val_list.append(df_x_val.drop(['CELLNAME'], axis=1).values)
    for drug in ['NSC1', 'NSC2']:
        for fea in self.drug_features:
            df_drug = getattr(self, self.drug_df_dict[fea])
            df_x_train = pd.merge(df_train[[drug]], df_drug, left_on=drug,
                right_on='NSC', how='left')
            df_x_val = pd.merge(df_val[[drug]], df_drug, left_on=drug,
                right_on='NSC', how='left')
            x_train_list.append(df_x_train.drop([drug, 'NSC'], axis=1).values)
            x_val_list.append(df_x_val.drop([drug, 'NSC'], axis=1).values)
    return x_train_list, y_train, x_val_list, y_val, df_train, df_val
