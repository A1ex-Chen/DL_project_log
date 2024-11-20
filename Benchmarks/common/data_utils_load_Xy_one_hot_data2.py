def load_Xy_one_hot_data2(train_file, test_file, class_col=None, drop_cols=
    None, n_cols=None, shuffle=False, scaling=None, validation_split=0.1,
    dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):
    """Load training and testing data from the files specified, with a column indicated to use as label.
    Further split trainig data into training and validation partitions,
    and construct corresponding training, validation and testing pandas DataFrames,
    separated into data (i.e. features) and labels. Labels to output are one-hot encoded (categorical).
    Columns to load can be selected or dropped. Order of rows
    can be shuffled. Data can be rescaled.
    Training and testing partitions (coming from the respective files)
    are preserved, but training is split into training and validation partitions.
    This function assumes that the files contain a header with column names.

    Parameters
    ----------
    train_file : filename
        Name of the file to load the training data.
    test_file : filename
        Name of the file to load the testing data.
    class_col : integer
        Index of the column to use as the label.
        (Default: None, this would cause the function to fail, a label
        has to be indicated at calling).
    drop_cols : list
        List of column names to drop from the files being loaded.
        (Default: None, all the columns are used).
    n_cols : integer
        Number of columns to load from the files.
        (Default: None, all the columns are used).
    shuffle : boolean
        Boolean flag to indicate row shuffling. If True the rows are
        re-ordered, if False the order in which rows are loaded is
        preserved.
        (Default: False, no permutation of the loading row order).
    scaling : string
        String describing type of scaling to apply.
        Options recognized: 'maxabs', 'minmax', 'std'.
        'maxabs' : scales data to range [-1 to 1].
        'minmax' : scales data to range [-1 to 1].
        'std'    : scales data to normal variable with mean 0 and standard deviation 1.
        (Default: None, no scaling).
    validation_split : float
        Fraction of training data to set aside for validation.
        (Default: 0.1, ten percent of the training data is
        used for the validation partition).
    dtype : data type
        Data type to use for the output pandas DataFrames.
        (Default: DEFAULT_DATATYPE defined in default_utils).
    seed : int
        Value to intialize or re-seed the generator.
        (Default: DEFAULT_SEED defined in default_utils).


    Return
    ----------
    X_train : pandas DataFrame
        Data features for training loaded in a pandas DataFrame and
        pre-processed as specified.
    y_train : pandas DataFrame
        Data labels for training loaded in a pandas DataFrame.
        One-hot encoding (categorical) is used.
    X_val : pandas DataFrame
        Data features for validation loaded in a pandas DataFrame and
        pre-processed as specified.
    y_val : pandas DataFrame
        Data labels for validation loaded in a pandas DataFrame.
        One-hot encoding (categorical) is used.
    X_test : pandas DataFrame
        Data features for testing loaded in a pandas DataFrame and
        pre-processed as specified.
    y_test : pandas DataFrame
        Data labels for testing loaded in a pandas DataFrame.
        One-hot encoding (categorical) is used.
    """
    assert class_col is not None
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None
    df_train = pd.read_csv(train_file, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_file, engine='c', usecols=usecols)
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)
    y_train = pd.get_dummies(df_train[class_col]).values
    y_test = pd.get_dummies(df_test[class_col]).values
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)
    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)
    mat = np.concatenate((X_train, X_test), axis=0)
    if scaling is not None:
        mat = scale_array(mat, scaling)
    sizeTrain = X_train.shape[0]
    X_test = mat[sizeTrain:, :]
    numVal = int(sizeTrain * validation_split)
    X_val = mat[:numVal, :]
    X_train = mat[numVal:sizeTrain, :]
    y_val = y_train[:numVal, :]
    y_train = y_train[numVal:sizeTrain, :]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
