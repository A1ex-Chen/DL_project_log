def generate_vocab(self):
    s = set(' ')
    for i, row in self.df.iterrows():
        s = s.union(row.iloc[0])
    print(s)
    self.vocab = list(s)
