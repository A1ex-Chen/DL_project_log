def load_data_by_index(self, train_index, val_index):
    x_all_list, y_all, df_all = self.load_data_all()
    x_train_list = [x[train_index] for x in x_all_list]
    x_val_list = [x[val_index] for x in x_all_list]
    y_train = y_all[train_index]
    y_val = y_all[val_index]
    df_train = df_all.iloc[train_index, :]
    df_val = df_all.iloc[val_index, :]
    if self.cv_partition == 'disjoint':
        logger.info('Training drugs: {}'.format(set(df_train['NSC1'])))
        logger.info('Validation drugs: {}'.format(set(df_val['NSC1'])))
    elif self.cv_partition == 'disjoint_cells':
        logger.info('Training cells: {}'.format(set(df_train['CELLNAME'])))
        logger.info('Validation cells: {}'.format(set(df_val['CELLNAME'])))
    return x_train_list, y_train, x_val_list, y_val, df_train, df_val
