def _get_response_on_memory(self, copy=False):
    if self.shuffle:
        self.index = [item for step in range(self.steps) for item in range(
            self.index_map[step] * self.batch_size, (self.index_map[step] +
            1) * self.batch_size)]
        df = self.dataframe['y_{}'.format(self.partition)].iloc[self.index, :]
    else:
        df = self.dataframe['y_{}'.format(self.partition)]
    if self.agg_dose is None:
        df['Dose1'] = self.dataframe['x_{}_0'.format(self.partition)].iloc[
            self.index, :]
        if not self.single:
            df['Dose2'] = self.dataframe['x_{}_1'.format(self.partition)].iloc[
                self.index, :]
    return df.copy() if copy else df
