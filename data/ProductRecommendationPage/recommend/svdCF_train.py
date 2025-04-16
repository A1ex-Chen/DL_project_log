def train(self, steps=2, gamma=0.04, Lambda=0.15):
    print('train data size', self.mat.shape)
    for step in range(steps):
        print('step', step + 1, 'is running')
        KK = np.random.permutation(self.mat.shape[0])
        rmse = 0.0
        for i in range(self.mat.shape[0]):
            j = KK[i]
            uid = self.mat[j, 0]
            iid = self.mat[j, 1]
            rating = self.mat[j, 2]
            eui = rating - self.predict_oneuser(uid, iid)
            rmse += eui ** 2
            self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
            self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
            tmp = self.qi[iid]
            self.qi[iid] += gamma * (eui * self.pu[uid] - Lambda * self.qi[iid]
                )
            self.pu[uid] += gamma * (eui * tmp - Lambda * self.pu[uid])
        gamma = 0.93 * gamma
        print('rmse is', np.sqrt(rmse / self.mat.shape[0]))
