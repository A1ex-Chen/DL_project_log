def modify_labels(numclasses_out, ytrain, ytest, yval=None):
    """This function generates a categorical representation with a class added for indicating abstention.

    Parameters
    ----------
    numclasses_out : integer
        Original number of classes + 1 abstention class
    ytrain : ndarray
        Numpy array of the classes (labels) in the training set
    ytest : ndarray
        Numpy array of the classes (labels) in the testing set
    yval : ndarray
        Numpy array of the classes (labels) in the validation set
    """
    classestrain = np.max(ytrain) + 1
    classestest = np.max(ytest) + 1
    if yval is not None:
        classesval = np.max(yval) + 1
    assert classestrain == classestest
    if yval is not None:
        assert classesval == classestest
    assert classestrain + 1 == numclasses_out
    labels_train = to_categorical(ytrain, numclasses_out)
    labels_test = to_categorical(ytest, numclasses_out)
    if yval is not None:
        labels_val = to_categorical(yval, numclasses_out)
    mask_vec = np.zeros(labels_train.shape)
    mask_vec[:, -1] = 1
    i = np.random.choice(range(labels_train.shape[0]))
    sanity_check = mask_vec[i, :] * labels_train[i, :]
    print(sanity_check.shape)
    if ytrain.ndim > 1:
        ll = ytrain.shape[1]
    else:
        ll = 0
    for i in range(ll):
        for j in range(numclasses_out):
            if sanity_check[i, j] == 1:
                print('Problem at ', i, j)
    if yval is not None:
        return labels_train, labels_test, labels_val
    return labels_train, labels_test
