def load_data(self):
    if self.cv_partition == 'disjoint':
        train_index = self.df_response[self.df_response['NSC1'].isin(self.
            train_drug_ids) & self.df_response['NSC2'].isin(self.
            train_drug_ids)].index
        val_index = self.df_response[self.df_response['NSC1'].isin(self.
            val_drug_ids) & self.df_response['NSC2'].isin(self.val_drug_ids)
            ].index
    else:
        train_index = range(self.n_train)
        val_index = range(self.n_train, self.total)
    return self.load_data_by_index(train_index, val_index)
