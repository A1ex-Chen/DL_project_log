def __init__(self, data, k=5, n=5):
    self.k = k
    self.n = n
    self.fn = self.pearson
    self.data = data
