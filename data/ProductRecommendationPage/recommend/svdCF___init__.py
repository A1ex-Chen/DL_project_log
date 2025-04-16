def __init__(self, df, K=20):
    self.df = df
    self.mat = np.array(df)
    self.K = K
    self.bi = {}
    self.bu = {}
    self.qi = {}
    self.pu = {}
    self.avg = np.mean(self.mat[:, 2])
    for i in range(self.mat.shape[0]):
        uid = self.mat[i, 0]
        iid = self.mat[i, 1]
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.
            sqrt(self.K))
        self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.
            sqrt(self.K))
