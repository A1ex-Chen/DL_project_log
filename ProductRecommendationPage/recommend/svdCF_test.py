def test(self, test_data):
    test_data = np.array(test_data)
    print('test data size', test_data.shape)
    rmse = 0.0
    for i in range(test_data.shape[0]):
        uid = test_data[i, 0]
        iid = test_data[i, 1]
        rating = test_data[i, 2]
        eui = rating - self.predict(uid, iid)
        rmse += eui ** 2
    print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))
