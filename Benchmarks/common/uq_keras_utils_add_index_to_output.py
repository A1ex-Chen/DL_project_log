def add_index_to_output(y_train):
    """This function adds a column to the training output to store the indices of the corresponding samples in the training set.

    Parameters
    ----------
    y_train : ndarray
        Numpy array of the output in the training set
    """
    y_train_index = range(y_train.shape[0])
    if y_train.ndim > 1:
        shp = y_train.shape[0], 1
        y_train_augmented = np.hstack([y_train, np.reshape(y_train_index, shp)]
            )
    else:
        y_train_augmented = np.vstack([y_train, y_train_index]).T
    return y_train_augmented
