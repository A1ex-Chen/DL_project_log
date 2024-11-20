def flow(self):
    """Keep generating data batches"""
    while 1:
        self.lock.acquire()
        indices = list(islice(self.cycle, self.batch_size))
        self.lock.release()
        df = self.data.df_response.iloc[indices, :]
        cell_column_beg = df.shape[1]
        for fea in self.data.cell_features:
            if fea == 'expression':
                df = pd.merge(df, self.data.df_cell_expr, on='CELLNAME')
            elif fea == 'mirna':
                df = pd.merge(df, self.data.df_cell_mirna, on='CELLNAME')
            elif fea == 'proteome':
                df = pd.merge(df, self.data.df_cell_prot, on='CELLNAME')
            elif fea == 'categorical':
                df = pd.merge(df, self.data.df_cell_cat, on='CELLNAME')
        cell_column_end = df.shape[1]
        for fea in self.data.drug_features:
            if fea == 'descriptors':
                df = df.merge(self.data.df_drug_desc, on='NSC')
            elif fea == 'latent':
                df = df.merge(self.data.df_drug_auen, on='NSC')
            elif fea == 'noise':
                df = df.merge(self.data.df_drug_rand, on='NSC')
        df = df.drop(['CELLNAME', 'NSC'], axis=1)
        x = np.array(df.iloc[:, 1:])
        if self.cell_noise_sigma:
            c1 = cell_column_beg - 3
            c2 = cell_column_end - 3
            x[:, c1:c2] += np.random.randn(df.shape[0], c2 - c1
                ) * self.cell_noise_sigma
        y = np.array(df.iloc[:, 0])
        y = y / 100.0
        if self.concat:
            if self.shape == 'add_1d':
                yield x.reshape(x.shape + (1,)), y
            else:
                yield x, y
        else:
            x_list = []
            index = 0
            for v in self.data.input_shapes.values():
                length = np.prod(v)
                subset = x[:, index:index + length]
                if self.shape == '1d':
                    reshape = x.shape[0], length
                elif self.shape == 'add_1d':
                    reshape = (x.shape[0],) + v + (1,)
                else:
                    reshape = (x.shape[0],) + v
                x_list.append(subset.reshape(reshape))
                index += length
            yield x_list, y
