def get_response(self, copy=False):
    df = self.df_data.iloc[self.index, :]
    return df.copy() if copy else df
