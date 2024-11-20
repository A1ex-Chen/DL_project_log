def make_implicit(self, df):
    print('Turning into implicit ratings')
    df = df[df['rating'] >= self.min_rating]
    return df
