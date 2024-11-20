def get_response(self, copy=False):
    df = self.data.df_response.iloc[self.index, :].drop(['Group'], axis=1)
    return df.copy() if copy else df
