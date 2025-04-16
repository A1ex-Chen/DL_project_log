def flow(self):
    """Keep generating data batches"""
    while 1:
        self.lock.acquire()
        indices = list(islice(self.cycle, self.batch_size))
        self.lock.release()
        df = self.data.df_response.iloc[indices, :]
        y = df['GROWTH'].values
        x_list = []
        for fea in self.data.cell_features:
            df_cell = getattr(self.data, self.data.cell_df_dict[fea])
            df_x = pd.merge(df[['CELLNAME']], df_cell, on='CELLNAME', how=
                'left')
            x_list.append(df_x.drop(['CELLNAME'], axis=1).values)
        for drug in ['NSC1', 'NSC2']:
            for fea in self.data.drug_features:
                df_drug = getattr(self.data, self.data.drug_df_dict[fea])
                df_x = pd.merge(df[[drug]], df_drug, left_on=drug, right_on
                    ='NSC', how='left')
                x_list.append(df_x.drop([drug, 'NSC'], axis=1).values)
        yield x_list, y
