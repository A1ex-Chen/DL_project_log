def interpolate_points(x, y, sampling):
    ln = LinearRegression()
    data = np.stack((x, y))
    data_train = np.array([0, 1]).reshape(-1, 1)
    ln.fit(data_train, data)
    return ln.predict(sampling.reshape(-1, 1)).astype(np.float32)
