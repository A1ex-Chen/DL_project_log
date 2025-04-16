def build_data(nnet_spec_len, fold, data_path):
    """Build feature sets to match the network topology"""
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(nnet_spec_len):
        feature_train = np.genfromtxt(data_path + '/task' + str(i) + '_' +
            str(fold) + '_train_feature.csv', delimiter=',')
        label_train = np.genfromtxt(data_path + '/task' + str(i) + '_' +
            str(fold) + '_train_label.csv', delimiter=',')
        X_train.append(feature_train)
        Y_train.append(label_train)
        feature_test = np.genfromtxt(data_path + '/task' + str(i) + '_' +
            str(fold) + '_test_feature.csv', delimiter=',')
        label_test = np.genfromtxt(data_path + '/task' + str(i) + '_' + str
            (fold) + '_test_label.csv', delimiter=',')
        X_test.append(feature_test)
        Y_test.append(label_test)
    return X_train, Y_train, X_test, Y_test
