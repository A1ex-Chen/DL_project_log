def load_X_data(train_file, test_file, drop_cols=None, n_cols=None, shuffle
    =False, scaling=None, dtype=DEFAULT_DATATYPE, seed=DEFAULT_SEED):
    """ Load training and testing unlabeleled data from the files specified
        and construct corresponding training and testing pandas DataFrames.
        Columns to load can be selected or dropped. Order of rows
        can be shuffled. Data can be rescaled.
        Training and testing partitions (coming from the respective files)
        are preserved.
        This function assumes that the files contain a header with column names.

        Parameters
        ----------
        train_file : filename
            Name of the file to load the training data.
        test_file : filename
            Name of the file to load the testing data.
        drop_cols : list
            List of column names to drop from the files being loaded.
            (Default: None, all the columns are used).
        n_cols : integer
            Number of columns to load from the files.
            (Default: None, all the columns are used).
        shuffle : boolean
            Boolean flag to indicate row shuffling. If True the rows are
            re-ordered, if False the order in which rows are read is
            preserved.
            (Default: False, no permutation of the loading row order).
        scaling : string
            String describing type of scaling to apply.
            Options recognized: 'maxabs', 'minmax', 'std'.
            'maxabs' : scales data to range [-1 to 1].
            'minmax' : scales data to range [-1 to 1].
            'std'    : scales data to normal variable with                        mean 0 and standard deviation 1.
            (Default: None, no scaling).
        dtype : data type
            Data type to use for the output pandas DataFrames.
            (Default: DEFAULT_DATATYPE defined in default_utils).
        seed : int
            Value to intialize or re-seed the generator.
            (Default: DEFAULT_SEED defined in default_utils).

        Return
        ----------
        X_train : pandas DataFrame
            Data for training loaded in a pandas DataFrame and pre-processed as specified.
        X_test : pandas DataFrame
            Data for testing loaded in a pandas DataFrame and pre-processed as specified.
    """
    usecols = list(range(n_cols + len(drop_cols))) if n_cols else None
    df_train = pd.read_csv(train_file, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_file, engine='c', usecols=usecols)
    if drop_cols is not None:
        for col in drop_cols:
            df_train.drop(col, axis=1, inplace=True)
            df_test.drop(col, axis=1, inplace=True)
    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)
    X_train = df_train.values.astype(dtype)
    X_test = df_test.values.astype(dtype)
    mat = np.concatenate((X_train, X_test), axis=0)
    if scaling is not None:
        mat = scale_array(mat, scaling)
    X_train = mat[:X_train.shape[0], :]
    X_test = mat[X_train.shape[0]:, :]
    return X_train, X_test
