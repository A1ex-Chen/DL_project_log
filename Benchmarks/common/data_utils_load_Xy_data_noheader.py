def load_Xy_data_noheader(train_file, test_file, classes, usecols=None,
    scaling=None, dtype=DEFAULT_DATATYPE):
    """Load training and testing data from the files specified, with the first column to use as label.
    Construct corresponding training and testing pandas DataFrames,
    separated into data (i.e. features) and labels.
    Labels to output are one-hot encoded (categorical).
    Columns to load can be selected. Data can be rescaled.
    Training and testing partitions (coming from the respective files)
    are preserved.
    This function assumes that the files do not contain a header.

    Parameters
    ----------
    train_file : filename
        Name of the file to load the training data.
    test_file : filename
        Name of the file to load the testing data.
    classes : integer
        Number of total classes to consider when
        building the categorical (one-hot) label encoding.
    usecols : list
        List of column indices to load from the files.
        (Default: None, all the columns are used).
    scaling : string
        String describing type of scaling to apply.
        Options recognized: 'maxabs', 'minmax', 'std'.
        'maxabs' : scales data to range [-1 to 1].
        'minmax' : scales data to range [-1 to 1].
        'std'    : scales data to normal variable with mean 0 and standard deviation 1.
        (Default: None, no scaling).
    dtype : data type
        Data type to use for the output pandas DataFrames.
        (Default: DEFAULT_DATATYPE defined in default_utils).

    Return
    ----------
    X_train : pandas DataFrame
        Data features for training loaded in a pandas DataFrame and
        pre-processed as specified.
    Y_train : pandas DataFrame
        Data labels for training loaded in a pandas DataFrame.
        One-hot encoding (categorical) is used.
    X_test : pandas DataFrame
        Data features for testing loaded in a pandas DataFrame and
        pre-processed as specified.
    Y_test : pandas DataFrame
        Data labels for testing loaded in a pandas DataFrame.
        One-hot encoding (categorical) is used.
    """
    print('Loading data...')
    df_train = pd.read_csv(train_file, header=None, usecols=usecols
        ).values.astype(dtype)
    df_test = pd.read_csv(test_file, header=None, usecols=usecols
        ).values.astype(dtype)
    print('done')
    print('df_train shape:', df_train.shape)
    print('df_test shape:', df_test.shape)
    seqlen = df_train.shape[1]
    df_y_train = df_train[:, 0].astype('int')
    df_y_test = df_test[:, 0].astype('int')
    Y_train = to_categorical(df_y_train, classes)
    Y_test = to_categorical(df_y_test, classes)
    df_x_train = df_train[:, 1:seqlen].astype(dtype)
    df_x_test = df_test[:, 1:seqlen].astype(dtype)
    X_train = df_x_train
    X_test = df_x_test
    mat = np.concatenate((X_train, X_test), axis=0)
    if scaling is not None:
        mat = scale_array(mat, scaling)
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]
    return X_train, Y_train, X_test, Y_test
