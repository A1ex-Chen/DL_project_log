def __init__(self, df, num=None):
    super(MoleLoader, self).__init__()
    size = df.shape[0]
    self.df = df.iloc[:int(size // 8), :]
    self.end_char = '?'
