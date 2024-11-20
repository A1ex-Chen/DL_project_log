def load_Xy_data2(train_file, test_file, class_col=None, drop_cols=None,
    n_cols=None, shuffle=False, scaling=None, validation_split=0.1, dtype=
    DEFAULT_DATATYPE, seed=DEFAULT_SEED):
    """Load training and testing data from the files specified, with a column indicated to use as label.
    Further split trainig data into training and validation partitions,
    and construct corresponding training, validation and testing pandas DataFrames,
    separated into data (i.e. features) and labels.
    Labels to output can be integer labels (for classification) or
    continuous labels (for regression).
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
    X_val : pandas DataFrame
        Data features for validation loaded in a pandas DataFrame and
        pre-processed as specified.
    y_val : pandas DataFrame
        Data labels for validation loaded in a pandas DataFrame.
    X_test : pandas DataFrame
        Data features for testing loaded in a pandas DataFrame and
        pre-processed as specified.
    y_test : pandas DataFrame
        Data labels for testing loaded in a pandas DataFrame.
    """
    assert class_col is not None
    (X_train, y_train_oh), (X_val, y_val_oh), (X_test, y_test_oh
        ) = load_Xy_one_hot_data2(train_file, test_file, class_col,
        drop_cols, n_cols, shuffle, scaling, validation_split, dtype, seed)
    y_train = convert_to_class(y_train_oh)
    y_val = convert_to_class(y_val_oh)
    y_test = convert_to_class(y_test_oh)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
