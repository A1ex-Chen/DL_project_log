def renormalize(self, X_train, mu, sigma):
    X_train = (X_train - mu) / sigma
    X_train = X_train.astype('float32')
    return X_train
