def add_noise(self, X_train):
    np.random.seed(100)
    ind = np.where(X_train == 0)
    rn = self.noise * np.random.rand(np.shape(ind)[1])
    X_train[ind] = rn
    return X_train
